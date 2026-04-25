"""RabbitMQ consumer that bridges the Java backend to the Python RAG.

Topology (declared idempotently on connect):
  Exchange `chat` (topic, durable)
  Queue `chat.search.request` ← bind `search.request`
  Queue `chat.search.reply`   ← bind `search.reply`

For each request: parse JSON, run async_pipeline_dispatch_chunks, publish
the JSON reply to routing key `search.reply` with header correlation_id.
"""
import asyncio
import json
import logging

import aio_pika

from src.config.app_config import get_app_config
from src.rag.search.pipeline_chunks import async_pipeline_dispatch_chunks
from src.utils.logger_utils import logger

EXCHANGE_NAME = "chat"
REQUEST_QUEUE = "chat.search.request"
REPLY_QUEUE = "chat.search.reply"
REQUEST_RK = "search.request"
REPLY_RK = "search.reply"
PREFETCH = 10

_INFLIGHT: set[asyncio.Task] = set()

# Suppress noisy aio_pika INFO logs
logging.getLogger("aio_pika").setLevel(logging.WARNING)
logging.getLogger("aiormq").setLevel(logging.WARNING)


async def _handle_message(
    message: aio_pika.IncomingMessage,
    exchange: aio_pika.abc.AbstractExchange,
) -> None:
    async with message.process(requeue=False):
        try:
            body = json.loads(message.body.decode("utf-8"))
        except Exception:
            logger.exception("Bad RabbitMQ payload, dropping")
            return

        correlation_id = body.get("correlationId") or message.correlation_id or ""
        user_message = body.get("userMessage", "")
        conversation_id = body.get("conversationId")

        logger.info(
            "rabbit search request | correlation_id=%s | conversation_id=%s",
            correlation_id, conversation_id,
        )

        try:
            reply = await async_pipeline_dispatch_chunks(user_message, conversation_id)
        except Exception as exc:
            logger.exception("async_pipeline_dispatch_chunks failed")
            reply = {
                "mode": "static",
                "intent": "off_topic",
                "detectedLanguage": "English",
                "standaloneQuery": user_message,
                "answerSatisfied": False,
                "webSearchUsed": False,
                "sources": [],
                "chunks": [],
                "response": "Internal error: the search service encountered an unexpected failure.",
            }

        reply["correlationId"] = correlation_id
        await exchange.publish(
            aio_pika.Message(
                body=json.dumps(reply, ensure_ascii=False).encode("utf-8"),
                content_type="application/json",
                correlation_id=correlation_id,
            ),
            routing_key=REPLY_RK,
        )


async def _consumer_loop() -> None:
    config = get_app_config()
    while True:
        try:
            connection = await aio_pika.connect_robust(config.RABBITMQ_URL)
            async with connection:
                channel = await connection.channel()
                await channel.set_qos(prefetch_count=PREFETCH)

                exchange = await channel.declare_exchange(
                    EXCHANGE_NAME, aio_pika.ExchangeType.TOPIC, durable=True,
                )
                request_queue = await channel.declare_queue(REQUEST_QUEUE, durable=True)
                reply_queue = await channel.declare_queue(REPLY_QUEUE, durable=True)
                await request_queue.bind(exchange, routing_key=REQUEST_RK)
                await reply_queue.bind(exchange, routing_key=REPLY_RK)

                logger.info("RabbitMQ consumer ready on %s", REQUEST_QUEUE)

                async with request_queue.iterator() as it:
                    async for message in it:
                        task = asyncio.create_task(_handle_message(message, exchange))
                        _INFLIGHT.add(task)
                        task.add_done_callback(_INFLIGHT.discard)
        except asyncio.CancelledError:
            logger.info("RabbitMQ consumer cancelled")
            raise
        except Exception:
            logger.exception("RabbitMQ consumer error — reconnecting in 5s")
            await asyncio.sleep(5)


def start_consumer() -> asyncio.Task:
    """Start the consumer as a background task."""
    return asyncio.create_task(_consumer_loop(), name="rabbit_consumer")
