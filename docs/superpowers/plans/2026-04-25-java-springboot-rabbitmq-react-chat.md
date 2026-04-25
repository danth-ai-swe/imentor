# Java Spring Boot + RabbitMQ + React Chat — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Java Spring Boot backend (REST + SSE + RabbitMQ + JPA) plus a React frontend that reuse the existing Python RAG retrieval. Java owns the OpenAI streaming call; Python only returns chunks.

**Architecture:** React → Java (SSE) → RabbitMQ → Python (existing). Python reuses `async_pipeline_dispatch` but skips answer generation, returning either pre-built static text or chunks. Java decides via the reply's `mode` field whether to stream a static reply or build a prompt and stream from OpenAI.

**Tech Stack:** Spring Boot 3.2 (Java 17, Maven), Spring Data JPA + H2 (file), Spring AMQP (RabbitMQ), WebClient (OpenAI streaming), Vite + React 18 + TypeScript, react-markdown, react-syntax-highlighter, aio-pika (Python).

**Spec:** `docs/superpowers/specs/2026-04-25-java-springboot-rabbitmq-react-chat-design.md`

**Project root:** `D:/Deverlopment/huudan.com/PythonProject` (existing). New folders: `backend/` (Java), `frontend/` (React). Python additions live under existing `src/`.

**Pre-flight checks:**
- RabbitMQ running on `localhost:5672` (guest/guest).
- `OPENAI_API_KEY` exported (used by both Python and Java).
- Python `.env` already exists with required vars (Qdrant, etc.).
- Java 17 + Maven installed (or use the wrapper that Task 4 generates).
- Node 18+ + npm installed.

---

## Phase 1 — Python additions

### Task 1: Add aio-pika dependency and RabbitMQ config

**Files:**
- Modify: `requirements.txt`
- Modify: `src/config/app_config.py`

- [ ] **Step 1: Append `aio-pika` to requirements.txt**

```
aio-pika
```

- [ ] **Step 2: Install the new dependency**

Run: `pip install aio-pika`
Expected: successful install of aio-pika and its `aiormq` transitive dep.

- [ ] **Step 3: Add `RABBITMQ_URL` to `AppConfig`**

In `src/config/app_config.py`, inside `class AppConfig(BaseSettings)`, after the existing string fields (e.g. after `SEARX_HOST`):

```python
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"
```

- [ ] **Step 4: Sanity-check the config loads**

Run: `python -c "from src.config.app_config import get_app_config; print(get_app_config().RABBITMQ_URL)"`
Expected: prints `amqp://guest:guest@localhost:5672/` (or the value from `.env` if overridden).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/config/app_config.py
git commit -m "feat(config): add aio-pika dep and RABBITMQ_URL setting"
```

---

### Task 2: Create chunks-only pipeline dispatcher

**Files:**
- Create: `src/rag/search/pipeline_chunks.py`
- Test: `tests/test_pipeline_chunks_shape.py`

- [ ] **Step 1: Write a shape test for the new dispatcher (off-topic branch — easiest to mock)**

Create `tests/test_pipeline_chunks_shape.py`:

```python
import asyncio
from unittest.mock import AsyncMock, patch

from src.rag.search.pipeline_chunks import async_pipeline_dispatch_chunks


def test_off_topic_returns_static_mode():
    async def run():
        with patch(
            "src.rag.search.pipeline_chunks._avalidate_and_prepare",
            new=AsyncMock(return_value=("English", "hi", None)),
        ), patch(
            "src.rag.search.pipeline_chunks.is_quiz_intent", return_value=False
        ), patch(
            "src.rag.search.pipeline_chunks._aroute_intent",
            new=AsyncMock(return_value="off_topic"),
        ), patch(
            "src.rag.search.pipeline_chunks.get_openai_chat_client", return_value=object()
        ), patch(
            "src.rag.search.pipeline_chunks.get_openai_embedding_client", return_value=object()
        ):
            result = await async_pipeline_dispatch_chunks("hi", None)

        assert result["mode"] == "static"
        assert result["intent"] == "off_topic"
        assert result["chunks"] == []
        assert result["response"]
        assert result["sources"] == []

    asyncio.run(run())
```

- [ ] **Step 2: Run the test to verify it fails (module not yet created)**

Run: `pytest tests/test_pipeline_chunks_shape.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.rag.search.pipeline_chunks'`.

- [ ] **Step 3: Create `src/rag/search/pipeline_chunks.py`**

```python
"""Chunks-only dispatcher for the Java backend.

Mirrors `async_pipeline_dispatch` from pipeline.py but skips the final
`_agenerate_answer` call. Returns either:
  - mode="static": a pre-built reply (off-topic, quiz, clarity, web fallback,
                   no-result, input-too-long); Java streams it as-is.
  - mode="chunks": enriched chunks + standalone_query + detected_language;
                   Java builds the prompt and streams from OpenAI.

Reuses the helpers in pipeline.py (validate, intent routing, clarity, HyDE,
embed, vector search, rerank, neighbor enrich) — does NOT duplicate them.
"""
from typing import Any, Dict, List, Optional

from src.apis.app_model import ChatSourceModel
from src.config.app_config import AppConfig, get_app_config
from src.constants.app_constant import (
    COLLECTION_NAME,
    OVERALL_COLLECTION_NAME,
    INTENT_CORE_KNOWLEDGE,
    INTENT_OFF_TOPIC,
    INTENT_QUIZ,
    INTENT_OVERALL_COURSE_KNOWLEDGE,
    OFF_TOPIC_RESPONSE_MAP,
    NO_RESULT_RESPONSE_MAP,
    CORE_VECTOR_TOP_K,
    CORE_RERANK_TOP_K,
    VECTOR_SEARCH_TOP_K,
)
from src.rag.db_vector import get_qdrant_client
from src.rag.llm.chat_llm import AzureChatClient, get_openai_chat_client
from src.rag.llm.embedding_llm import AzureEmbeddingClient, get_openai_embedding_client
from src.rag.search.pipeline import (
    _acheck_input_clarity,
    _aembed_text,
    _afetch_neighbor_chunks,
    _ahyde_generate,
    _aroute_intent,
    _avalidate_and_prepare,
    _avector_search,
    _merge_chunks,
    extract_sources,
)
from src.rag.search.reranker import arerank_chunks
from src.rag.search.searxng_search import web_rag_answer
from src.utils.app_utils import is_quiz_intent
from src.utils.logger_utils import logger, StepTimer


def _serialize_sources(sources: List[Any]) -> List[Dict[str, Any]]:
    return [
        s.model_dump(by_alias=False) if isinstance(s, ChatSourceModel) else dict(s)
        for s in (sources or [])
    ]


def _serialize_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in chunks or []:
        out.append({
            "text": c.get("text", ""),
            "metadata": dict(c.get("metadata", {})),
        })
    return out


def _static(
    *, intent: str, response: Optional[str], detected_language: Optional[str],
    sources: Optional[List[Any]] = None, web_search_used: bool = False,
    answer_satisfied: bool = True, standalone_query: str = "",
) -> Dict[str, Any]:
    return {
        "mode": "static",
        "intent": intent,
        "detectedLanguage": detected_language,
        "standaloneQuery": standalone_query,
        "answerSatisfied": answer_satisfied,
        "webSearchUsed": web_search_used,
        "sources": _serialize_sources(sources or []),
        "chunks": [],
        "response": response or "",
    }


def _chunks_payload(
    *, intent: str, detected_language: str, standalone_query: str,
    sources: List[Any], chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "mode": "chunks",
        "intent": intent,
        "detectedLanguage": detected_language,
        "standaloneQuery": standalone_query,
        "answerSatisfied": True,
        "webSearchUsed": False,
        "sources": _serialize_sources(sources),
        "chunks": _serialize_chunks(chunks),
        "response": None,
    }


async def _arun_core_search_chunks(
    llm: AzureChatClient,
    embedder: AzureEmbeddingClient,
    standalone_query: str,
    detected_language: str,
) -> Dict[str, Any]:
    timer = StepTimer(f"core_search_chunks:{COLLECTION_NAME}")
    config: AppConfig = get_app_config()

    try:
        qdrant = get_qdrant_client(COLLECTION_NAME)

        async with timer.astep("clarity_check"):
            clarity = await _acheck_input_clarity(llm, standalone_query, detected_language)
        if not clarity.get("clear", True):
            response_parts = clarity.get("response", [])
            response_str = (
                response_parts if isinstance(response_parts, str)
                else "\n".join(r.strip() for r in response_parts if r) if isinstance(response_parts, list)
                else str(response_parts)
            )
            return _static(
                intent=INTENT_CORE_KNOWLEDGE, response=response_str,
                detected_language=detected_language, answer_satisfied=False,
                standalone_query=standalone_query,
            )

        async with timer.astep("hyde"):
            hyde = await _ahyde_generate(llm, standalone_query)
        async with timer.astep("hyde_embedding"):
            dense_vec, colbert_vec = await _aembed_text(embedder, qdrant, hyde)
        async with timer.astep("vector_search"):
            sorted_chunks = await _avector_search(
                qdrant, dense_vec, colbert_vec, standalone_query, top_k=CORE_VECTOR_TOP_K,
            )

        if not sorted_chunks:
            async with timer.astep("web_search_fallback"):
                rs = await web_rag_answer(llm, embedder, standalone_query, detected_language)
            return _static(
                intent=INTENT_CORE_KNOWLEDGE, response=rs["answer"],
                detected_language=detected_language, sources=rs["sources"],
                web_search_used=True, standalone_query=standalone_query,
            )

        async with timer.astep("rerank"):
            reranked = await arerank_chunks(standalone_query, sorted_chunks, CORE_RERANK_TOP_K)
        async with timer.astep("enrich_chunks"):
            neighbor_chunks = await _afetch_neighbor_chunks(qdrant, reranked)
            enriched_chunks = _merge_chunks(reranked, neighbor_chunks)

        sources = extract_sources(reranked, config.APP_DOMAIN)
        return _chunks_payload(
            intent=INTENT_CORE_KNOWLEDGE, detected_language=detected_language,
            standalone_query=standalone_query, sources=sources, chunks=enriched_chunks,
        )
    finally:
        timer.summary()


async def _arun_overall_search_chunks(
    llm: AzureChatClient,
    embedder: AzureEmbeddingClient,
    standalone_query: str,
    detected_language: str,
) -> Dict[str, Any]:
    timer = StepTimer(f"overall_search_chunks:{OVERALL_COLLECTION_NAME}")

    try:
        qdrant = get_qdrant_client(OVERALL_COLLECTION_NAME)

        async with timer.astep("clarity_check"):
            clarity = await _acheck_input_clarity(llm, standalone_query, detected_language)
        if not clarity.get("clear", True):
            response_parts = clarity.get("response", [])
            response_str = (
                response_parts if isinstance(response_parts, str)
                else "\n".join(r.strip() for r in response_parts if r) if isinstance(response_parts, list)
                else str(response_parts)
            )
            return _static(
                intent=INTENT_OVERALL_COURSE_KNOWLEDGE, response=response_str,
                detected_language=detected_language, answer_satisfied=False,
                standalone_query=standalone_query,
            )

        async with timer.astep("query_embedding"):
            dense_vec, colbert_vec = await _aembed_text(embedder, qdrant, standalone_query)
        async with timer.astep("vector_search"):
            sorted_chunks = await _avector_search(
                qdrant, dense_vec, colbert_vec, standalone_query, top_k=VECTOR_SEARCH_TOP_K,
            )

        if not sorted_chunks:
            no_result = NO_RESULT_RESPONSE_MAP.get(detected_language) or ""
            return _static(
                intent=INTENT_OVERALL_COURSE_KNOWLEDGE, response=no_result,
                detected_language=detected_language, answer_satisfied=False,
                standalone_query=standalone_query,
            )

        return _chunks_payload(
            intent=INTENT_OVERALL_COURSE_KNOWLEDGE, detected_language=detected_language,
            standalone_query=standalone_query, sources=[], chunks=sorted_chunks,
        )
    finally:
        timer.summary()


async def async_pipeline_dispatch_chunks(
    user_input: str,
    conversation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Chunks-only equivalent of async_pipeline_dispatch. See module docstring."""
    timer = StepTimer("dispatch_chunks")
    try:
        llm = get_openai_chat_client()
        embedder = get_openai_embedding_client()

        async with timer.astep("validate_and_prepare"):
            detected_language, standalone_query, error_response = (
                await _avalidate_and_prepare(user_input, conversation_id, llm)
            )
            if error_response:
                return _static(
                    intent=INTENT_OFF_TOPIC, response=error_response,
                    detected_language="English", standalone_query=user_input,
                    answer_satisfied=False,
                )

        async with timer.astep("quiz_keyword_check"):
            if is_quiz_intent(user_input) or is_quiz_intent(standalone_query):
                return _static(
                    intent=INTENT_QUIZ,
                    response="Quiz feature is not enabled in this demo.",
                    detected_language=detected_language, answer_satisfied=False,
                    standalone_query=standalone_query,
                )

        async with timer.astep("intent_routing"):
            intent = await _aroute_intent(embedder, standalone_query)

        if intent == INTENT_OFF_TOPIC:
            return _static(
                intent=INTENT_OFF_TOPIC,
                response=OFF_TOPIC_RESPONSE_MAP.get(detected_language) or "",
                detected_language=detected_language, answer_satisfied=False,
                standalone_query=standalone_query,
            )

        if intent == INTENT_OVERALL_COURSE_KNOWLEDGE:
            return await _arun_overall_search_chunks(llm, embedder, standalone_query, detected_language)

        return await _arun_core_search_chunks(llm, embedder, standalone_query, detected_language)
    finally:
        timer.summary()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_pipeline_chunks_shape.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag/search/pipeline_chunks.py tests/test_pipeline_chunks_shape.py
git commit -m "feat(rag): chunks-only dispatcher for external orchestration"
```

---

### Task 3: RabbitMQ consumer + lifespan integration

**Files:**
- Create: `src/messaging/__init__.py`
- Create: `src/messaging/rabbit_consumer.py`
- Modify: `main.py`

- [ ] **Step 1: Create empty package marker**

Create `src/messaging/__init__.py` (empty file).

- [ ] **Step 2: Create the consumer module**

Create `src/messaging/rabbit_consumer.py`:

```python
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
                "response": f"Internal error: {exc}",
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
                        asyncio.create_task(_handle_message(message, exchange))
        except asyncio.CancelledError:
            logger.info("RabbitMQ consumer cancelled")
            raise
        except Exception:
            logger.exception("RabbitMQ consumer error — reconnecting in 5s")
            await asyncio.sleep(5)


def start_consumer() -> asyncio.Task:
    """Start the consumer as a background task."""
    return asyncio.create_task(_consumer_loop(), name="rabbit_consumer")
```

- [ ] **Step 3: Wire the consumer into the FastAPI lifespan**

Edit `main.py`. Add the import near the top with the other src imports:

```python
from src.messaging.rabbit_consumer import start_consumer
```

Inside `lifespan`, after the line `asyncio.get_running_loop().run_in_executor(None, _get_reranker)` and before `yield`, add:

```python
    consumer_task = start_consumer()
```

After `yield`, before `get_langfuse_client().flush()`, add:

```python
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
```

- [ ] **Step 4: Verify the consumer starts (manual smoke)**

Run: `python main.py` (in a terminal with RabbitMQ running)
Expected: among the startup logs, see `RabbitMQ consumer ready on chat.search.request`. Stop with Ctrl+C — should see `RabbitMQ consumer cancelled`.

- [ ] **Step 5: Commit**

```bash
git add src/messaging/__init__.py src/messaging/rabbit_consumer.py main.py
git commit -m "feat(messaging): rabbit consumer bridging Java backend to RAG"
```

---

## Phase 2 — Java backend scaffolding + domain

### Task 4: Maven scaffolding

**Files:**
- Create: `backend/pom.xml`
- Create: `backend/.gitignore`
- Create: `backend/src/main/java/com/example/chat/ChatApplication.java`
- Create: `backend/src/main/resources/application.yml`
- Create: `backend/mvnw`, `backend/mvnw.cmd`, `backend/.mvn/wrapper/maven-wrapper.properties`

- [ ] **Step 1: Create the backend folder + pom.xml**

Create `backend/pom.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
    <modelVersion>4.0.0</modelVersion>

    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.2.5</version>
        <relativePath/>
    </parent>

    <groupId>com.example</groupId>
    <artifactId>chat-backend</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <packaging>jar</packaging>

    <properties>
        <java.version>17</java.version>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-webflux</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-amqp</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-jpa</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-validation</artifactId>
        </dependency>
        <dependency>
            <groupId>com.h2database</groupId>
            <artifactId>h2</artifactId>
            <scope>runtime</scope>
        </dependency>
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <optional>true</optional>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
                <configuration>
                    <excludes>
                        <exclude>
                            <groupId>org.projectlombok</groupId>
                            <artifactId>lombok</artifactId>
                        </exclude>
                    </excludes>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

- [ ] **Step 2: Create the Spring Boot main class**

Create `backend/src/main/java/com/example/chat/ChatApplication.java`:

```java
package com.example.chat;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class ChatApplication {
    public static void main(String[] args) {
        SpringApplication.run(ChatApplication.class, args);
    }
}
```

- [ ] **Step 3: Create application.yml**

Create `backend/src/main/resources/application.yml`:

```yaml
server:
  port: 8080
spring:
  datasource:
    url: jdbc:h2:file:./data/chatdb;AUTO_SERVER=TRUE
    driverClassName: org.h2.Driver
    username: sa
    password:
  jpa:
    hibernate:
      ddl-auto: update
    show-sql: false
  rabbitmq:
    host: localhost
    port: 5672
    username: guest
    password: guest
  h2:
    console:
      enabled: true
      path: /h2-console
openai:
  api-key: ${OPENAI_API_KEY:}
  base-url: https://api.openai.com
  model: gpt-4o-mini
chat:
  search-timeout-seconds: 30
logging:
  level:
    com.example.chat: INFO
```

- [ ] **Step 4: Create backend `.gitignore`**

Create `backend/.gitignore`:

```
target/
data/
.idea/
*.iml
.vscode/
HELP.md
```

- [ ] **Step 5: Generate the Maven wrapper**

Run (from `backend/`): `mvn -N wrapper:wrapper -Dmaven=3.9.6`
Expected: `mvnw`, `mvnw.cmd`, and `.mvn/wrapper/maven-wrapper.properties` are created.

(If `mvn` is not installed system-wide, copy these files from any other Spring Boot project on the machine, or run `mvnw -N wrapper:wrapper` once a wrapper is present.)

- [ ] **Step 6: Verify the app boots**

Run (from `backend/`): `./mvnw spring-boot:run`
Expected: Spring Boot startup logs end with `Tomcat started on port 8080`. Stop with Ctrl+C.

- [ ] **Step 7: Commit**

```bash
git add backend/pom.xml backend/.gitignore backend/.mvn backend/mvnw backend/mvnw.cmd \
        backend/src/main/java/com/example/chat/ChatApplication.java \
        backend/src/main/resources/application.yml
git commit -m "feat(backend): scaffold Spring Boot + Maven + H2 + RabbitMQ deps"
```

---

### Task 5: JPA entities + repositories

**Files:**
- Create: `backend/src/main/java/com/example/chat/domain/User.java`
- Create: `backend/src/main/java/com/example/chat/domain/Conversation.java`
- Create: `backend/src/main/java/com/example/chat/domain/Message.java`
- Create: `backend/src/main/java/com/example/chat/domain/ChunkSource.java`
- Create: `backend/src/main/java/com/example/chat/repository/UserRepository.java`
- Create: `backend/src/main/java/com/example/chat/repository/ConversationRepository.java`
- Create: `backend/src/main/java/com/example/chat/repository/MessageRepository.java`
- Create: `backend/src/main/java/com/example/chat/repository/ChunkSourceRepository.java`

- [ ] **Step 1: Create `User.java`**

```java
package com.example.chat.domain;

import jakarta.persistence.*;
import lombok.*;

import java.time.Instant;

@Entity
@Table(name = "users", uniqueConstraints = @UniqueConstraint(columnNames = "username"))
@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class User {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String username;

    @Column(nullable = false)
    private Instant createdAt;

    @PrePersist
    void prePersist() {
        if (createdAt == null) createdAt = Instant.now();
    }
}
```

- [ ] **Step 2: Create `Conversation.java`**

```java
package com.example.chat.domain;

import jakarta.persistence.*;
import lombok.*;

import java.time.Instant;

@Entity
@Table(name = "conversations")
@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class Conversation {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "user_id")
    private User user;

    private String title;

    @Column(nullable = false)
    private Instant createdAt;

    @Column(nullable = false)
    private Instant updatedAt;

    @PrePersist
    void prePersist() {
        Instant now = Instant.now();
        if (createdAt == null) createdAt = now;
        updatedAt = now;
    }

    @PreUpdate
    void preUpdate() {
        updatedAt = Instant.now();
    }
}
```

- [ ] **Step 3: Create `Message.java`**

```java
package com.example.chat.domain;

import jakarta.persistence.*;
import lombok.*;

import java.time.Instant;

@Entity
@Table(name = "messages")
@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class Message {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "conversation_id")
    private Conversation conversation;

    @Column(nullable = false, length = 16)
    private String role; // "user" or "assistant"

    @Lob
    @Column(columnDefinition = "CLOB")
    private String content;

    private String intent;
    private String detectedLanguage;
    private boolean webSearchUsed;
    private boolean stopped;

    @Column(nullable = false)
    private Instant createdAt;

    @PrePersist
    void prePersist() {
        if (createdAt == null) createdAt = Instant.now();
    }
}
```

- [ ] **Step 4: Create `ChunkSource.java`**

```java
package com.example.chat.domain;

import jakarta.persistence.*;
import lombok.*;

@Entity
@Table(name = "chunk_sources")
@Getter @Setter @NoArgsConstructor @AllArgsConstructor @Builder
public class ChunkSource {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "message_id")
    private Message message;

    private String name;

    @Column(length = 1024)
    private String url;

    private Integer pageNumber;
    private Integer totalPages;
}
```

- [ ] **Step 5: Create the four repositories**

`UserRepository.java`:

```java
package com.example.chat.repository;

import com.example.chat.domain.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Long> {
    Optional<User> findByUsername(String username);
}
```

`ConversationRepository.java`:

```java
package com.example.chat.repository;

import com.example.chat.domain.Conversation;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface ConversationRepository extends JpaRepository<Conversation, Long> {
    List<Conversation> findByUserUsernameOrderByUpdatedAtDesc(String username);
}
```

`MessageRepository.java`:

```java
package com.example.chat.repository;

import com.example.chat.domain.Message;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface MessageRepository extends JpaRepository<Message, Long> {
    List<Message> findByConversationIdOrderByCreatedAtAsc(Long conversationId);
    List<Message> findByConversationIdAndIdGreaterThanEqual(Long conversationId, Long messageId);
}
```

`ChunkSourceRepository.java`:

```java
package com.example.chat.repository;

import com.example.chat.domain.ChunkSource;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface ChunkSourceRepository extends JpaRepository<ChunkSource, Long> {
    List<ChunkSource> findByMessageId(Long messageId);
    void deleteByMessageIdIn(List<Long> messageIds);
}
```

- [ ] **Step 6: Verify compile + boot**

Run (from `backend/`): `./mvnw spring-boot:run`
Expected: startup logs include `Hibernate: create table users ...` (and other tables) on first run, ends with Tomcat on 8080. Stop with Ctrl+C.

- [ ] **Step 7: Commit**

```bash
git add backend/src/main/java/com/example/chat/domain backend/src/main/java/com/example/chat/repository
git commit -m "feat(backend): JPA entities + repositories for chat persistence"
```

---

### Task 6: DTOs

**Files:**
- Create: `backend/src/main/java/com/example/chat/dto/CreateUserRequest.java`
- Create: `backend/src/main/java/com/example/chat/dto/UserDto.java`
- Create: `backend/src/main/java/com/example/chat/dto/CreateConversationRequest.java`
- Create: `backend/src/main/java/com/example/chat/dto/ConversationDto.java`
- Create: `backend/src/main/java/com/example/chat/dto/SourceDto.java`
- Create: `backend/src/main/java/com/example/chat/dto/MessageDto.java`
- Create: `backend/src/main/java/com/example/chat/dto/AskRequest.java`
- Create: `backend/src/main/java/com/example/chat/dto/ChunkDto.java`
- Create: `backend/src/main/java/com/example/chat/dto/SearchRequestMessage.java`
- Create: `backend/src/main/java/com/example/chat/dto/SearchReplyMessage.java`

- [ ] **Step 1: Request/response DTOs**

`CreateUserRequest.java`:

```java
package com.example.chat.dto;

import jakarta.validation.constraints.NotBlank;

public record CreateUserRequest(@NotBlank String username) {}
```

`UserDto.java`:

```java
package com.example.chat.dto;

public record UserDto(Long id, String username) {}
```

`CreateConversationRequest.java`:

```java
package com.example.chat.dto;

import jakarta.validation.constraints.NotNull;

public record CreateConversationRequest(@NotNull Long userId, String title) {}
```

`ConversationDto.java`:

```java
package com.example.chat.dto;

import java.time.Instant;

public record ConversationDto(Long id, String title, Instant createdAt, Instant updatedAt) {}
```

`SourceDto.java`:

```java
package com.example.chat.dto;

public record SourceDto(String name, String url, Integer pageNumber, Integer totalPages) {}
```

`MessageDto.java`:

```java
package com.example.chat.dto;

import java.time.Instant;
import java.util.List;

public record MessageDto(
    Long id,
    String role,
    String content,
    String intent,
    String detectedLanguage,
    boolean webSearchUsed,
    boolean stopped,
    Instant createdAt,
    List<SourceDto> sources
) {}
```

`AskRequest.java`:

```java
package com.example.chat.dto;

import jakarta.validation.constraints.NotBlank;

public record AskRequest(@NotBlank String message) {}
```

- [ ] **Step 2: RabbitMQ message DTOs**

`ChunkDto.java`:

```java
package com.example.chat.dto;

import java.util.Map;

public record ChunkDto(String text, Map<String, Object> metadata) {}
```

`SearchRequestMessage.java`:

```java
package com.example.chat.dto;

public record SearchRequestMessage(String correlationId, String userMessage, String conversationId) {}
```

`SearchReplyMessage.java`:

```java
package com.example.chat.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record SearchReplyMessage(
    String correlationId,
    String mode,                // "chunks" | "static"
    String intent,
    String detectedLanguage,
    String standaloneQuery,
    boolean answerSatisfied,
    boolean webSearchUsed,
    List<SourceDto> sources,
    List<ChunkDto> chunks,
    String response
) {}
```

- [ ] **Step 3: Verify compile**

Run (from `backend/`): `./mvnw -q compile`
Expected: BUILD SUCCESS.

- [ ] **Step 4: Commit**

```bash
git add backend/src/main/java/com/example/chat/dto
git commit -m "feat(backend): DTOs for REST and RabbitMQ contracts"
```

---

## Phase 3 — Java integrations (Rabbit + OpenAI)

### Task 7: RabbitMQ config + RPC client + reply listener

**Files:**
- Create: `backend/src/main/java/com/example/chat/config/RabbitConfig.java`
- Create: `backend/src/main/java/com/example/chat/messaging/SearchRpcClient.java`
- Create: `backend/src/main/java/com/example/chat/messaging/SearchReplyListener.java`

- [ ] **Step 1: Rabbit config**

`config/RabbitConfig.java`:

```java
package com.example.chat.config;

import org.springframework.amqp.core.*;
import org.springframework.amqp.rabbit.connection.ConnectionFactory;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.amqp.support.converter.Jackson2JsonMessageConverter;
import org.springframework.amqp.support.converter.MessageConverter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class RabbitConfig {

    public static final String EXCHANGE = "chat";
    public static final String REQUEST_QUEUE = "chat.search.request";
    public static final String REPLY_QUEUE = "chat.search.reply";
    public static final String REQUEST_RK = "search.request";
    public static final String REPLY_RK = "search.reply";

    @Bean public TopicExchange chatExchange()       { return ExchangeBuilder.topicExchange(EXCHANGE).durable(true).build(); }
    @Bean public Queue searchRequestQueue()         { return QueueBuilder.durable(REQUEST_QUEUE).build(); }
    @Bean public Queue searchReplyQueue()           { return QueueBuilder.durable(REPLY_QUEUE).build(); }
    @Bean public Binding requestBinding()           { return BindingBuilder.bind(searchRequestQueue()).to(chatExchange()).with(REQUEST_RK); }
    @Bean public Binding replyBinding()             { return BindingBuilder.bind(searchReplyQueue()).to(chatExchange()).with(REPLY_RK); }

    @Bean
    public MessageConverter jsonMessageConverter() {
        return new Jackson2JsonMessageConverter();
    }

    @Bean
    public RabbitTemplate rabbitTemplate(ConnectionFactory cf, MessageConverter conv) {
        RabbitTemplate t = new RabbitTemplate(cf);
        t.setMessageConverter(conv);
        t.setExchange(EXCHANGE);
        return t;
    }
}
```

- [ ] **Step 2: SearchRpcClient**

`messaging/SearchRpcClient.java`:

```java
package com.example.chat.messaging;

import com.example.chat.config.RabbitConfig;
import com.example.chat.dto.SearchReplyMessage;
import com.example.chat.dto.SearchRequestMessage;
import org.springframework.amqp.core.MessageProperties;
import org.springframework.amqp.rabbit.core.RabbitTemplate;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.UUID;
import java.util.concurrent.*;

@Component
public class SearchRpcClient {

    private final RabbitTemplate rabbit;
    private final long timeoutSeconds;
    private final ConcurrentHashMap<String, CompletableFuture<SearchReplyMessage>> pending = new ConcurrentHashMap<>();

    public SearchRpcClient(
        RabbitTemplate rabbit,
        @Value("${chat.search-timeout-seconds:30}") long timeoutSeconds
    ) {
        this.rabbit = rabbit;
        this.timeoutSeconds = timeoutSeconds;
    }

    public SearchReplyMessage requestSearch(String userMessage, String conversationId) {
        String correlationId = UUID.randomUUID().toString();
        CompletableFuture<SearchReplyMessage> future = new CompletableFuture<>();
        pending.put(correlationId, future);

        SearchRequestMessage payload = new SearchRequestMessage(correlationId, userMessage, conversationId);
        rabbit.convertAndSend(RabbitConfig.EXCHANGE, RabbitConfig.REQUEST_RK, payload, m -> {
            m.getMessageProperties().setCorrelationId(correlationId);
            m.getMessageProperties().setContentType(MessageProperties.CONTENT_TYPE_JSON);
            return m;
        });

        try {
            return future.get(timeoutSeconds, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            pending.remove(correlationId);
            throw new RuntimeException("Search request timed out after " + timeoutSeconds + "s", e);
        } catch (InterruptedException | ExecutionException e) {
            pending.remove(correlationId);
            throw new RuntimeException("Search request failed", e);
        }
    }

    void complete(String correlationId, SearchReplyMessage reply) {
        CompletableFuture<SearchReplyMessage> f = pending.remove(correlationId);
        if (f != null) f.complete(reply);
    }
}
```

- [ ] **Step 3: SearchReplyListener**

`messaging/SearchReplyListener.java`:

```java
package com.example.chat.messaging;

import com.example.chat.dto.SearchReplyMessage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.amqp.rabbit.annotation.RabbitListener;
import org.springframework.stereotype.Component;

@Component
public class SearchReplyListener {
    private static final Logger log = LoggerFactory.getLogger(SearchReplyListener.class);
    private final SearchRpcClient client;

    public SearchReplyListener(SearchRpcClient client) {
        this.client = client;
    }

    @RabbitListener(queues = "chat.search.reply")
    public void onReply(SearchReplyMessage reply) {
        if (reply == null || reply.correlationId() == null) {
            log.warn("Reply with no correlationId, dropping");
            return;
        }
        client.complete(reply.correlationId(), reply);
    }
}
```

- [ ] **Step 4: Verify boot — Rabbit topology declared**

Pre-requisite: RabbitMQ + the existing Python consumer should NOT be required for this step — Java will declare the topology on its own.

Run (from `backend/`): `./mvnw spring-boot:run`
Expected: startup logs include `Created new connection: rabbitConnectionFactory#...` and `(Created queue) chat.search.reply` (or no errors at minimum). Stop with Ctrl+C.

In another shell, verify queues exist: open `http://localhost:15672` (RabbitMQ management UI, guest/guest) → Queues tab → see `chat.search.request` and `chat.search.reply`.

- [ ] **Step 5: Commit**

```bash
git add backend/src/main/java/com/example/chat/config/RabbitConfig.java \
        backend/src/main/java/com/example/chat/messaging
git commit -m "feat(backend): RabbitMQ config + RPC client/listener for search"
```

---

### Task 8: OpenAI streaming service + PromptBuilder + unit test

**Files:**
- Create: `backend/src/main/java/com/example/chat/config/OpenAiConfig.java`
- Create: `backend/src/main/java/com/example/chat/openai/OpenAiChatService.java`
- Create: `backend/src/main/java/com/example/chat/openai/PromptBuilder.java`
- Create: `backend/src/test/java/com/example/chat/openai/PromptBuilderTest.java`

- [ ] **Step 1: OpenAi WebClient config**

`config/OpenAiConfig.java`:

```java
package com.example.chat.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.http.client.HttpClient;

import java.time.Duration;

@Configuration
public class OpenAiConfig {

    @Bean(name = "openAiWebClient")
    public WebClient openAiWebClient(
        @Value("${openai.base-url}") String baseUrl,
        @Value("${openai.api-key}") String apiKey
    ) {
        HttpClient httpClient = HttpClient.create().responseTimeout(Duration.ofSeconds(120));
        return WebClient.builder()
            .baseUrl(baseUrl)
            .defaultHeader(HttpHeaders.AUTHORIZATION, "Bearer " + apiKey)
            .defaultHeader(HttpHeaders.ACCEPT, MediaType.TEXT_EVENT_STREAM_VALUE)
            .clientConnector(new ReactorClientHttpConnector(httpClient))
            .build();
    }
}
```

- [ ] **Step 2: PromptBuilder**

`openai/PromptBuilder.java`:

```java
package com.example.chat.openai;

import com.example.chat.dto.ChunkDto;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

@Component
public class PromptBuilder {

    private static final String TEMPLATE = """
        You are **Insuripedia** — a warm, witty LOMA281/LOMA291 study buddy.
        Answer using ONLY the Retrieved Context. Reply in **%s**.
        Be concise. Open with one warm sentence + a fitting emoji. **Bold** key terms.
        Use bullets for 3+ items. Never invent facts. Never end with a question.

        ## Retrieved Context
        %s

        ## User Question
        %s
        """;

    public String build(List<ChunkDto> chunks, String standaloneQuery, String detectedLanguage) {
        String language = (detectedLanguage == null || detectedLanguage.isBlank()) ? "English" : detectedLanguage;
        String contextStr = chunks == null || chunks.isEmpty()
            ? "(no context)"
            : chunks.stream().map(this::renderChunk).reduce((a, b) -> a + "\n\n---\n\n" + b).orElse("");
        return TEMPLATE.formatted(language, contextStr, standaloneQuery == null ? "" : standaloneQuery);
    }

    private String renderChunk(ChunkDto c) {
        Map<String, Object> meta = c.metadata() == null ? Map.of() : c.metadata();
        String fileName = String.valueOf(meta.getOrDefault("file_name", "unknown"));
        StringBuilder header = new StringBuilder("Source: ").append(fileName);
        for (String key : new String[]{"course", "module_number", "lesson_number", "page_number"}) {
            Object v = meta.get(key);
            if (v != null && !String.valueOf(v).isEmpty()) {
                header.append(" | ").append(prettyKey(key)).append(": ").append(v);
            }
        }
        return "[" + header + "]\n" + (c.text() == null ? "" : c.text());
    }

    private String prettyKey(String k) {
        return switch (k) {
            case "module_number" -> "Module";
            case "lesson_number" -> "Lesson";
            case "page_number"   -> "Page";
            default              -> Character.toUpperCase(k.charAt(0)) + k.substring(1);
        };
    }
}
```

- [ ] **Step 3: Write the failing PromptBuilder test**

`backend/src/test/java/com/example/chat/openai/PromptBuilderTest.java`:

```java
package com.example.chat.openai;

import com.example.chat.dto.ChunkDto;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class PromptBuilderTest {

    private final PromptBuilder builder = new PromptBuilder();

    @Test
    void buildsPromptWithLanguageContextAndQuery() {
        ChunkDto chunk = new ChunkDto(
            "Adverse selection occurs when...",
            Map.of("file_name", "loma281.pdf", "page_number", 12)
        );
        String prompt = builder.build(List.of(chunk), "What is adverse selection?", "Vietnamese");
        assertThat(prompt)
            .contains("Reply in **Vietnamese**")
            .contains("Source: loma281.pdf | Page: 12")
            .contains("Adverse selection occurs when...")
            .contains("## User Question")
            .contains("What is adverse selection?");
    }

    @Test
    void usesPlaceholderWhenNoChunks() {
        String prompt = builder.build(List.of(), "Hi", null);
        assertThat(prompt).contains("(no context)").contains("Reply in **English**");
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run (from `backend/`): `./mvnw -q test -Dtest=PromptBuilderTest`
Expected: tests PASS.

- [ ] **Step 5: OpenAiChatService streaming**

`openai/OpenAiChatService.java`:

```java
package com.example.chat.openai;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.Map;

@Service
public class OpenAiChatService {

    private static final Logger log = LoggerFactory.getLogger(OpenAiChatService.class);
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final WebClient webClient;
    private final String model;

    public OpenAiChatService(
        @Qualifier("openAiWebClient") WebClient webClient,
        @Value("${openai.model}") String model
    ) {
        this.webClient = webClient;
        this.model = model;
    }

    public Flux<String> streamChat(String prompt) {
        Map<String, Object> body = Map.of(
            "model", model,
            "stream", true,
            "messages", List.of(Map.of("role", "user", "content", prompt))
        );

        return webClient.post()
            .uri("/v1/chat/completions")
            .contentType(MediaType.APPLICATION_JSON)
            .bodyValue(body)
            .retrieve()
            .bodyToFlux(String.class) // each "data: {...}" SSE event arrives as a String
            .filter(line -> line != null && !line.isBlank())
            .takeWhile(line -> !"[DONE]".equals(line.trim()))
            .mapNotNull(this::extractContent)
            .filter(s -> !s.isEmpty());
    }

    private String extractContent(String dataLine) {
        try {
            JsonNode root = MAPPER.readTree(dataLine);
            JsonNode delta = root.path("choices").path(0).path("delta").path("content");
            return delta.isMissingNode() || delta.isNull() ? "" : delta.asText();
        } catch (Exception e) {
            log.warn("Failed to parse OpenAI SSE chunk: {}", dataLine);
            return "";
        }
    }
}
```

- [ ] **Step 6: Verify compile**

Run (from `backend/`): `./mvnw -q compile`
Expected: BUILD SUCCESS.

- [ ] **Step 7: Commit**

```bash
git add backend/src/main/java/com/example/chat/config/OpenAiConfig.java \
        backend/src/main/java/com/example/chat/openai \
        backend/src/test/java/com/example/chat/openai
git commit -m "feat(backend): OpenAI streaming WebClient + PromptBuilder + test"
```

---

### Task 9: CORS for the React dev server

**Files:**
- Create: `backend/src/main/java/com/example/chat/config/WebConfig.java`

- [ ] **Step 1: Create WebConfig**

```java
package com.example.chat.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/api/**")
            .allowedOrigins("http://localhost:5173")
            .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")
            .allowedHeaders("*");
    }
}
```

- [ ] **Step 2: Verify boot**

Run (from `backend/`): `./mvnw -q spring-boot:run` → Ctrl+C after Tomcat starts.

- [ ] **Step 3: Commit**

```bash
git add backend/src/main/java/com/example/chat/config/WebConfig.java
git commit -m "feat(backend): CORS config for React dev origin"
```

---

## Phase 4 — Java REST + SSE

### Task 10: User/conversation/messages CRUD endpoints

**Files:**
- Create: `backend/src/main/java/com/example/chat/chat/ChatService.java`
- Create: `backend/src/main/java/com/example/chat/chat/ChatController.java`

- [ ] **Step 1: ChatService — CRUD methods first**

`chat/ChatService.java`:

```java
package com.example.chat.chat;

import com.example.chat.domain.*;
import com.example.chat.dto.*;
import com.example.chat.repository.*;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
public class ChatService {

    private final UserRepository users;
    private final ConversationRepository conversations;
    private final MessageRepository messages;
    private final ChunkSourceRepository chunkSources;

    public ChatService(UserRepository users, ConversationRepository conversations,
                       MessageRepository messages, ChunkSourceRepository chunkSources) {
        this.users = users;
        this.conversations = conversations;
        this.messages = messages;
        this.chunkSources = chunkSources;
    }

    @Transactional
    public UserDto getOrCreateUser(String username) {
        User u = users.findByUsername(username)
            .orElseGet(() -> users.save(User.builder().username(username).build()));
        return new UserDto(u.getId(), u.getUsername());
    }

    @Transactional(readOnly = true)
    public List<ConversationDto> listConversations(String username) {
        return conversations.findByUserUsernameOrderByUpdatedAtDesc(username).stream()
            .map(c -> new ConversationDto(c.getId(), c.getTitle(), c.getCreatedAt(), c.getUpdatedAt()))
            .toList();
    }

    @Transactional
    public ConversationDto createConversation(Long userId, String title) {
        User user = users.findById(userId).orElseThrow(() -> new IllegalArgumentException("user not found"));
        Conversation c = conversations.save(Conversation.builder()
            .user(user)
            .title(title == null || title.isBlank() ? "New chat" : title)
            .build());
        return new ConversationDto(c.getId(), c.getTitle(), c.getCreatedAt(), c.getUpdatedAt());
    }

    @Transactional(readOnly = true)
    public List<MessageDto> listMessages(Long conversationId) {
        return messages.findByConversationIdOrderByCreatedAtAsc(conversationId).stream()
            .map(m -> {
                List<SourceDto> sources = chunkSources.findByMessageId(m.getId()).stream()
                    .map(s -> new SourceDto(s.getName(), s.getUrl(), s.getPageNumber(), s.getTotalPages()))
                    .toList();
                return new MessageDto(
                    m.getId(), m.getRole(), m.getContent(), m.getIntent(),
                    m.getDetectedLanguage(), m.isWebSearchUsed(), m.isStopped(),
                    m.getCreatedAt(), sources
                );
            }).toList();
    }

    @Transactional
    public void deleteFromMessage(Long conversationId, Long messageId) {
        List<Message> toDelete = messages.findByConversationIdAndIdGreaterThanEqual(conversationId, messageId);
        if (toDelete.isEmpty()) return;
        List<Long> ids = toDelete.stream().map(Message::getId).toList();
        chunkSources.deleteByMessageIdIn(ids);
        messages.deleteAll(toDelete);
    }
}
```

- [ ] **Step 2: ChatController — CRUD endpoints (no SSE yet)**

`chat/ChatController.java`:

```java
package com.example.chat.chat;

import com.example.chat.dto.*;
import jakarta.validation.Valid;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api")
public class ChatController {

    private final ChatService service;

    public ChatController(ChatService service) {
        this.service = service;
    }

    @PostMapping("/users")
    public UserDto createUser(@Valid @RequestBody CreateUserRequest req) {
        return service.getOrCreateUser(req.username());
    }

    @GetMapping("/users/{username}/conversations")
    public List<ConversationDto> listConversations(@PathVariable String username) {
        return service.listConversations(username);
    }

    @PostMapping("/conversations")
    public ConversationDto createConversation(@Valid @RequestBody CreateConversationRequest req) {
        return service.createConversation(req.userId(), req.title());
    }

    @GetMapping("/conversations/{id}/messages")
    public List<MessageDto> listMessages(@PathVariable Long id) {
        return service.listMessages(id);
    }

    @DeleteMapping("/conversations/{convId}/messages/from/{messageId}")
    public ResponseEntity<Void> deleteFromMessage(@PathVariable Long convId, @PathVariable Long messageId) {
        service.deleteFromMessage(convId, messageId);
        return ResponseEntity.noContent().build();
    }
}
```

- [ ] **Step 3: Smoke-test the CRUD endpoints with curl**

Start the backend: `./mvnw spring-boot:run` (in another terminal).

Then:

```bash
# Create user
curl -s -X POST http://localhost:8080/api/users -H "Content-Type: application/json" -d '{"username":"alice"}'
# Expected: {"id":1,"username":"alice"}

# Create conversation
curl -s -X POST http://localhost:8080/api/conversations -H "Content-Type: application/json" -d '{"userId":1,"title":"Chat 1"}'
# Expected: {"id":1,"title":"Chat 1","createdAt":"...","updatedAt":"..."}

# List conversations
curl -s http://localhost:8080/api/users/alice/conversations
# Expected: [{"id":1,"title":"Chat 1",...}]

# List messages (should be empty)
curl -s http://localhost:8080/api/conversations/1/messages
# Expected: []
```

Stop the server with Ctrl+C.

- [ ] **Step 4: Commit**

```bash
git add backend/src/main/java/com/example/chat/chat
git commit -m "feat(backend): user/conversation/message CRUD endpoints"
```

---

### Task 11: SSE `/ask/stream` endpoint

**Files:**
- Modify: `backend/src/main/java/com/example/chat/chat/ChatService.java`
- Modify: `backend/src/main/java/com/example/chat/chat/ChatController.java`

- [ ] **Step 1: Add streaming method to `ChatService`**

In `ChatService.java`, add the imports at top:

```java
import com.example.chat.messaging.SearchRpcClient;
import com.example.chat.openai.OpenAiChatService;
import com.example.chat.openai.PromptBuilder;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
import reactor.core.Disposable;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
```

Add fields + extend constructor:

```java
    private static final Logger log = LoggerFactory.getLogger(ChatService.class);
    private static final ObjectMapper JSON = new ObjectMapper();

    private final SearchRpcClient rpc;
    private final OpenAiChatService openai;
    private final PromptBuilder promptBuilder;
```

Replace the constructor with:

```java
    public ChatService(UserRepository users, ConversationRepository conversations,
                       MessageRepository messages, ChunkSourceRepository chunkSources,
                       SearchRpcClient rpc, OpenAiChatService openai, PromptBuilder promptBuilder) {
        this.users = users;
        this.conversations = conversations;
        this.messages = messages;
        this.chunkSources = chunkSources;
        this.rpc = rpc;
        this.openai = openai;
        this.promptBuilder = promptBuilder;
    }
```

Add the streaming method:

```java
    public SseEmitter ask(Long conversationId, String userMessage) {
        SseEmitter emitter = new SseEmitter(60_000L);

        Conversation conv = conversations.findById(conversationId)
            .orElseThrow(() -> new IllegalArgumentException("conversation not found"));

        // 1) Persist the user message
        Message userMsg = messages.save(Message.builder()
            .conversation(conv).role("user").content(userMessage).build());
        conversations.save(conv); // touches updatedAt

        // 2) RPC → Python
        SearchReplyMessage reply;
        try {
            reply = rpc.requestSearch(userMessage, String.valueOf(conversationId));
        } catch (Exception ex) {
            sendEvent(emitter, "error", Map.of("message", ex.getMessage()));
            emitter.complete();
            return emitter;
        }

        // 3) Persist empty assistant message
        Message assistantMsg = messages.save(Message.builder()
            .conversation(conv).role("assistant").content("")
            .intent(reply.intent())
            .detectedLanguage(reply.detectedLanguage())
            .webSearchUsed(reply.webSearchUsed())
            .build());

        // 4) Persist chunk sources (if any)
        if (reply.sources() != null) {
            for (SourceDto s : reply.sources()) {
                chunkSources.save(ChunkSource.builder()
                    .message(assistantMsg)
                    .name(s.name()).url(s.url())
                    .pageNumber(s.pageNumber()).totalPages(s.totalPages())
                    .build());
            }
        }

        // 5) Emit meta event
        sendEvent(emitter, "meta", Map.of(
            "intent", nullSafe(reply.intent()),
            "detectedLanguage", nullSafe(reply.detectedLanguage()),
            "sources", reply.sources() == null ? java.util.List.of() : reply.sources(),
            "webSearchUsed", reply.webSearchUsed(),
            "assistantMessageId", assistantMsg.getId()
        ));

        // 6) Branch on mode
        if ("static".equals(reply.mode())) {
            String text = reply.response() == null ? "" : reply.response();
            sendEvent(emitter, "delta", Map.of("content", text));
            assistantMsg.setContent(text);
            messages.save(assistantMsg);
            sendEvent(emitter, "done", Map.of());
            emitter.complete();
            return emitter;
        }

        // mode == "chunks": stream from OpenAI
        StringBuilder buffer = new StringBuilder();
        String prompt = promptBuilder.build(reply.chunks(), reply.standaloneQuery(), reply.detectedLanguage());

        AtomicReference<Disposable> subRef = new AtomicReference<>();
        Disposable sub = openai.streamChat(prompt).subscribe(
            token -> {
                buffer.append(token);
                sendEvent(emitter, "delta", Map.of("content", token));
            },
            error -> {
                log.warn("OpenAI stream error", error);
                sendEvent(emitter, "error", Map.of("message", error.getMessage()));
                persistAssistant(assistantMsg, buffer.toString(), buffer.length() > 0);
                emitter.complete();
            },
            () -> {
                persistAssistant(assistantMsg, buffer.toString(), false);
                sendEvent(emitter, "done", Map.of());
                emitter.complete();
            }
        );
        subRef.set(sub);

        emitter.onTimeout(() -> { sub.dispose(); persistAssistant(assistantMsg, buffer.toString(), buffer.length() > 0); });
        emitter.onError(e -> {
            sub.dispose();
            persistAssistant(assistantMsg, buffer.toString(), buffer.length() > 0);
        });
        emitter.onCompletion(() -> {
            if (!sub.isDisposed()) sub.dispose();
        });

        return emitter;
    }

    @Transactional
    void persistAssistant(Message m, String content, boolean stopped) {
        m.setContent(content);
        m.setStopped(stopped);
        messages.save(m);
    }

    private static String nullSafe(String s) { return s == null ? "" : s; }

    private void sendEvent(SseEmitter emitter, String name, Object data) {
        try {
            emitter.send(SseEmitter.event().name(name).data(JSON.writeValueAsString(data)));
        } catch (IOException e) {
            log.debug("SSE send failed (client disconnected): {}", e.getMessage());
        }
    }
```

- [ ] **Step 2: Add the controller endpoint**

In `ChatController.java`, add imports:

```java
import org.springframework.http.MediaType;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;
```

Add method:

```java
    @PostMapping(path = "/conversations/{id}/ask/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter ask(@PathVariable Long id, @Valid @RequestBody AskRequest req) {
        return service.ask(id, req.message());
    }
```

- [ ] **Step 3: End-to-end smoke test**

Pre-requisites: RabbitMQ + Python (`python main.py`) + `OPENAI_API_KEY` exported.

Start Java: `./mvnw spring-boot:run`

In another shell:

```bash
curl -N -X POST http://localhost:8080/api/conversations/1/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"What is adverse selection?"}'
```

Expected: `event: meta\ndata: {...}` followed by multiple `event: delta\ndata: {"content":"..."}` lines, ending with `event: done\ndata: {}`. Actual answer text should be about adverse selection in insurance.

- [ ] **Step 4: Commit**

```bash
git add backend/src/main/java/com/example/chat/chat
git commit -m "feat(backend): /ask/stream SSE endpoint with rabbit RPC + OpenAI"
```

---

## Phase 5 — React frontend

### Task 12: Vite scaffolding

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/.gitignore`

- [ ] **Step 1: package.json**

```json
{
  "name": "chat-frontend",
  "private": true,
  "version": "0.0.1",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-markdown": "^9.0.1",
    "remark-gfm": "^4.0.0",
    "react-syntax-highlighter": "^15.5.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@types/react-syntax-highlighter": "^15.5.13",
    "@vitejs/plugin-react": "^4.3.1",
    "typescript": "^5.4.5",
    "vite": "^5.3.1"
  }
}
```

- [ ] **Step 2: tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"]
}
```

- [ ] **Step 3: vite.config.ts**

```ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8080',
    },
  },
});
```

- [ ] **Step 4: index.html**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Insuripedia Chat</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 5: src/main.tsx**

```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './App.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

- [ ] **Step 6: .gitignore**

```
node_modules/
dist/
.vite/
```

- [ ] **Step 7: Install dependencies**

Run (from `frontend/`): `npm install`
Expected: `node_modules/` created, no errors.

- [ ] **Step 8: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/tsconfig.json \
        frontend/vite.config.ts frontend/index.html frontend/src/main.tsx frontend/.gitignore
git commit -m "feat(frontend): vite + react + typescript scaffolding"
```

---

### Task 13: Types + API client

**Files:**
- Create: `frontend/src/types.ts`
- Create: `frontend/src/api/chatApi.ts`

- [ ] **Step 1: types.ts**

```ts
export interface User { id: number; username: string; }
export interface Conversation { id: number; title: string; createdAt: string; updatedAt: string; }
export interface Source { name: string; url: string; pageNumber: number; totalPages: number; }
export interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  intent?: string;
  detectedLanguage?: string;
  webSearchUsed?: boolean;
  stopped?: boolean;
  createdAt: string;
  sources: Source[];
}

export type SseHandler = {
  onMeta: (meta: { intent: string; detectedLanguage: string; sources: Source[];
                   webSearchUsed: boolean; assistantMessageId: number }) => void;
  onDelta: (token: string) => void;
  onDone: () => void;
  onError: (err: string) => void;
};
```

- [ ] **Step 2: chatApi.ts**

```ts
import type { Conversation, Message, SseHandler, User } from '../types';

const BASE = '/api';

export async function createUser(username: string): Promise<User> {
  const res = await fetch(`${BASE}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username }),
  });
  return res.json();
}

export async function listConversations(username: string): Promise<Conversation[]> {
  const res = await fetch(`${BASE}/users/${encodeURIComponent(username)}/conversations`);
  return res.json();
}

export async function createConversation(userId: number, title?: string): Promise<Conversation> {
  const res = await fetch(`${BASE}/conversations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ userId, title }),
  });
  return res.json();
}

export async function loadMessages(conversationId: number): Promise<Message[]> {
  const res = await fetch(`${BASE}/conversations/${conversationId}/messages`);
  return res.json();
}

export async function deleteFromMessage(conversationId: number, messageId: number): Promise<void> {
  await fetch(`${BASE}/conversations/${conversationId}/messages/from/${messageId}`, { method: 'DELETE' });
}

export async function askStream(
  conversationId: number,
  message: string,
  handlers: SseHandler,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch(`${BASE}/conversations/${conversationId}/ask/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({ message }),
    signal,
  });
  if (!res.body) throw new Error('No response body');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEvent = 'message';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let nl;
      while ((nl = buffer.indexOf('\n')) !== -1) {
        const rawLine = buffer.slice(0, nl).replace(/\r$/, '');
        buffer = buffer.slice(nl + 1);

        if (rawLine === '') { currentEvent = 'message'; continue; }
        if (rawLine.startsWith('event:')) { currentEvent = rawLine.slice(6).trim(); continue; }
        if (rawLine.startsWith('data:')) {
          const data = rawLine.slice(5).trim();
          if (!data) continue;
          try {
            const parsed = JSON.parse(data);
            if (currentEvent === 'meta') handlers.onMeta(parsed);
            else if (currentEvent === 'delta') handlers.onDelta(parsed.content ?? '');
            else if (currentEvent === 'done') handlers.onDone();
            else if (currentEvent === 'error') handlers.onError(parsed.message ?? 'unknown error');
          } catch { /* skip malformed line */ }
        }
      }
    }
  } catch (e: any) {
    if (e.name !== 'AbortError') handlers.onError(String(e.message ?? e));
  }
}
```

- [ ] **Step 3: TypeScript check**

Run (from `frontend/`): `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/types.ts frontend/src/api/chatApi.ts
git commit -m "feat(frontend): types + REST/SSE API client"
```

---

### Task 14: useChatStream hook

**Files:**
- Create: `frontend/src/hooks/useChatStream.ts`

- [ ] **Step 1: Create the hook**

```ts
import { useRef, useState } from 'react';
import { askStream } from '../api/chatApi';
import type { Message, Source } from '../types';

export function useChatStream() {
  const [streaming, setStreaming] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  async function start(
    conversationId: number,
    userMessage: string,
    onAppendUser: (msg: Message) => void,
    onPlaceholder: (placeholder: Message) => void,
    onUpdatePlaceholder: (
      assistantMessageId: number,
      patch: { content?: string; sources?: Source[]; intent?: string;
               detectedLanguage?: string; webSearchUsed?: boolean },
    ) => void,
    onError: (err: string) => void,
  ) {
    if (streaming) return;
    setStreaming(true);

    const tempUserId = -Date.now();
    onAppendUser({
      id: tempUserId, role: 'user', content: userMessage,
      createdAt: new Date().toISOString(), sources: [],
    });

    const placeholderId = -(Date.now() + 1);
    let buffer = '';
    onPlaceholder({
      id: placeholderId, role: 'assistant', content: '',
      createdAt: new Date().toISOString(), sources: [],
    });

    const controller = new AbortController();
    controllerRef.current = controller;

    try {
      await askStream(conversationId, userMessage, {
        onMeta: (meta) => {
          onUpdatePlaceholder(placeholderId, {
            sources: meta.sources, intent: meta.intent,
            detectedLanguage: meta.detectedLanguage, webSearchUsed: meta.webSearchUsed,
          });
        },
        onDelta: (tok) => {
          buffer += tok;
          onUpdatePlaceholder(placeholderId, { content: buffer });
        },
        onDone: () => { /* nothing extra — caller will reload from server */ },
        onError: (msg) => onError(msg),
      }, controller.signal);
    } finally {
      setStreaming(false);
      controllerRef.current = null;
    }
  }

  function stop() {
    controllerRef.current?.abort();
  }

  return { streaming, start, stop };
}
```

- [ ] **Step 2: TypeScript check**

Run (from `frontend/`): `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/hooks/useChatStream.ts
git commit -m "feat(frontend): useChatStream hook with abort support"
```

---

### Task 15: MessageBubble + SuggestedPrompts components

**Files:**
- Create: `frontend/src/components/MessageBubble.tsx`
- Create: `frontend/src/components/SuggestedPrompts.tsx`

- [ ] **Step 1: MessageBubble.tsx**

```tsx
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { Message } from '../types';

interface Props {
  message: Message;
  streaming: boolean;
  isLastAssistant: boolean;
  onEdit?: () => void;
  onRegenerate?: () => void;
}

export function MessageBubble({ message, streaming, isLastAssistant, onEdit, onRegenerate }: Props) {
  const isUser = message.role === 'user';
  return (
    <div className={`bubble ${isUser ? 'user' : 'assistant'}`}>
      <div className="bubble-body">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            code({ inline, className, children, ...props }: any) {
              const match = /language-(\w+)/.exec(className || '');
              if (!inline && match) {
                return (
                  <SyntaxHighlighter
                    style={oneDark as any}
                    language={match[1]}
                    PreTag="div"
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                );
              }
              return <code className={className} {...props}>{children}</code>;
            },
          }}
        >
          {message.content || ''}
        </ReactMarkdown>
        {streaming && !isUser && <span className="cursor">▋</span>}
      </div>
      {!isUser && message.sources && message.sources.length > 0 && (
        <div className="sources">
          <strong>Sources:</strong>
          <ul>
            {message.sources.map((s, i) => (
              <li key={i}>
                <a href={s.url} target="_blank" rel="noreferrer">
                  {s.name} (p.{s.pageNumber}/{s.totalPages})
                </a>
              </li>
            ))}
          </ul>
        </div>
      )}
      {message.stopped && <div className="stopped-marker">(stopped)</div>}
      {!streaming && (
        <div className="bubble-actions">
          {isUser && onEdit && <button onClick={onEdit}>✎ Edit</button>}
          {!isUser && isLastAssistant && onRegenerate && (
            <button onClick={onRegenerate}>↻ Regenerate</button>
          )}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: SuggestedPrompts.tsx**

```tsx
const SUGGESTED_PROMPTS = [
  'What is adverse selection in insurance?',
  'Explain the law of large numbers.',
  'What is reinsurance and why is it important?',
  'List the main types of insurance contracts.',
];

export function SuggestedPrompts({ onPick }: { onPick: (prompt: string) => void }) {
  return (
    <div className="suggested-prompts">
      <h3>Try asking…</h3>
      <div className="prompts-grid">
        {SUGGESTED_PROMPTS.map((p) => (
          <button key={p} className="prompt-card" onClick={() => onPick(p)}>{p}</button>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: TypeScript check**

Run (from `frontend/`): `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/MessageBubble.tsx frontend/src/components/SuggestedPrompts.tsx
git commit -m "feat(frontend): message bubble (markdown + highlight) + suggested prompts"
```

---

### Task 16: ChatWindow component (input, send/stop, edit, regenerate)

**Files:**
- Create: `frontend/src/components/ChatWindow.tsx`

- [ ] **Step 1: Create ChatWindow.tsx**

```tsx
import { useEffect, useRef, useState } from 'react';
import { deleteFromMessage, loadMessages } from '../api/chatApi';
import { useChatStream } from '../hooks/useChatStream';
import type { Conversation, Message, Source } from '../types';
import { MessageBubble } from './MessageBubble';
import { SuggestedPrompts } from './SuggestedPrompts';

interface Props { conversation: Conversation; }

export function ChatWindow({ conversation }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editValue, setEditValue] = useState('');
  const [error, setError] = useState<string | null>(null);
  const { streaming, start, stop } = useChatStream();
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadMessages(conversation.id).then(setMessages);
  }, [conversation.id]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
  }, [messages]);

  function appendUser(msg: Message) { setMessages((m) => [...m, msg]); }
  function appendPlaceholder(p: Message) { setMessages((m) => [...m, p]); }
  function patchPlaceholder(
    placeholderId: number,
    patch: { content?: string; sources?: Source[]; intent?: string;
             detectedLanguage?: string; webSearchUsed?: boolean },
  ) {
    setMessages((m) => m.map((x) => (x.id === placeholderId ? { ...x, ...patch } as Message : x)));
  }

  async function send(text: string) {
    setError(null);
    setInput('');
    await start(conversation.id, text, appendUser, appendPlaceholder, patchPlaceholder, setError);
    // After streaming completes, reload to pull real DB ids.
    const fresh = await loadMessages(conversation.id);
    setMessages(fresh);
  }

  async function regenerate(assistantMsg: Message) {
    const idx = messages.findIndex((m) => m.id === assistantMsg.id);
    if (idx < 1) return;
    const previousUser = messages[idx - 1];
    if (previousUser.role !== 'user') return;
    await deleteFromMessage(conversation.id, assistantMsg.id);
    setMessages((m) => m.filter((x) => x.id !== assistantMsg.id));
    await send(previousUser.content);
  }

  async function commitEdit(userMsg: Message, newContent: string) {
    setEditingId(null);
    await deleteFromMessage(conversation.id, userMsg.id);
    setMessages((m) => m.filter((x) => x.id < userMsg.id));
    await send(newContent);
  }

  const lastAssistantId = [...messages].reverse().find((m) => m.role === 'assistant')?.id ?? -1;

  return (
    <div className="chat-window">
      <header className="chat-header">{conversation.title}</header>
      <div className="messages" ref={listRef}>
        {messages.length === 0 && !streaming && <SuggestedPrompts onPick={send} />}
        {messages.map((m) => {
          if (editingId === m.id) {
            return (
              <div key={m.id} className="bubble user editing">
                <textarea value={editValue} onChange={(e) => setEditValue(e.target.value)} />
                <div className="edit-actions">
                  <button onClick={() => commitEdit(m, editValue)}>Save</button>
                  <button onClick={() => setEditingId(null)}>Cancel</button>
                </div>
              </div>
            );
          }
          const isPlaceholder = m.id < 0 && m.role === 'assistant';
          return (
            <MessageBubble
              key={m.id}
              message={m}
              streaming={streaming && isPlaceholder}
              isLastAssistant={m.id === lastAssistantId && !streaming}
              onEdit={m.role === 'user' ? () => { setEditingId(m.id); setEditValue(m.content); } : undefined}
              onRegenerate={m.role === 'assistant' ? () => regenerate(m) : undefined}
            />
          );
        })}
      </div>
      {error && <div className="error">{error}</div>}
      <form className="composer" onSubmit={(e) => { e.preventDefault(); if (input.trim()) send(input.trim()); }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask anything…"
          rows={2}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (input.trim()) send(input.trim()); }
          }}
        />
        {streaming
          ? <button type="button" className="stop-btn" onClick={stop}>■ Stop</button>
          : <button type="submit" disabled={!input.trim()}>Send</button>}
      </form>
    </div>
  );
}
```

- [ ] **Step 2: TypeScript check**

Run (from `frontend/`): `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ChatWindow.tsx
git commit -m "feat(frontend): ChatWindow with input/send/stop/edit/regenerate"
```

---

### Task 17: Sidebar + App + CSS

**Files:**
- Create: `frontend/src/components/Sidebar.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/App.css`

- [ ] **Step 1: Sidebar.tsx**

```tsx
import { useEffect, useState } from 'react';
import { createConversation, listConversations } from '../api/chatApi';
import type { Conversation, User } from '../types';

interface Props {
  user: User;
  selectedId: number | null;
  onSelect: (c: Conversation) => void;
  refreshKey: number;
}

export function Sidebar({ user, selectedId, onSelect, refreshKey }: Props) {
  const [items, setItems] = useState<Conversation[]>([]);

  useEffect(() => {
    listConversations(user.username).then(setItems);
  }, [user.username, refreshKey]);

  async function newChat() {
    const c = await createConversation(user.id, 'New chat');
    setItems((s) => [c, ...s]);
    onSelect(c);
  }

  return (
    <aside className="sidebar">
      <button className="new-chat-btn" onClick={newChat}>+ New Chat</button>
      <div className="conversations">
        {items.map((c) => (
          <button
            key={c.id}
            className={`conv-item ${c.id === selectedId ? 'selected' : ''}`}
            onClick={() => onSelect(c)}
          >
            <div className="conv-title">{c.title || 'Untitled'}</div>
            <div className="conv-date">{new Date(c.updatedAt).toLocaleString()}</div>
          </button>
        ))}
      </div>
      <div className="user-footer">Signed in as <strong>{user.username}</strong></div>
    </aside>
  );
}
```

- [ ] **Step 2: App.tsx**

```tsx
import { useEffect, useState } from 'react';
import { createUser } from './api/chatApi';
import { ChatWindow } from './components/ChatWindow';
import { Sidebar } from './components/Sidebar';
import type { Conversation, User } from './types';

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [pendingName, setPendingName] = useState('');
  const [active, setActive] = useState<Conversation | null>(null);
  const [refreshKey] = useState(0);

  useEffect(() => {
    const raw = localStorage.getItem('chat.user');
    if (raw) setUser(JSON.parse(raw));
  }, []);

  async function signIn() {
    const u = await createUser(pendingName.trim());
    localStorage.setItem('chat.user', JSON.stringify(u));
    setUser(u);
  }

  if (!user) {
    return (
      <div className="signin">
        <h2>Welcome</h2>
        <input
          autoFocus placeholder="Enter your name"
          value={pendingName} onChange={(e) => setPendingName(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && pendingName.trim()) signIn(); }}
        />
        <button disabled={!pendingName.trim()} onClick={signIn}>Continue</button>
      </div>
    );
  }

  return (
    <div className="app">
      <Sidebar user={user} selectedId={active?.id ?? null} onSelect={setActive} refreshKey={refreshKey} />
      <main className="main">
        {active
          ? <ChatWindow key={active.id} conversation={active} />
          : <div className="empty-state">Pick a conversation or start a new one.</div>}
      </main>
    </div>
  );
}
```

- [ ] **Step 3: App.css**

```css
* { box-sizing: border-box; }
html, body, #root { height: 100%; margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }

.signin { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; gap: 12px; }
.signin input { padding: 8px 12px; font-size: 16px; }
.signin button { padding: 8px 16px; font-size: 16px; cursor: pointer; }

.app { display: grid; grid-template-columns: 280px 1fr; height: 100vh; }

.sidebar { background: #f4f5f7; border-right: 1px solid #e1e4e8; display: flex; flex-direction: column; padding: 12px; gap: 8px; }
.new-chat-btn { padding: 10px; font-size: 14px; cursor: pointer; background: #2d6cdf; color: white; border: none; border-radius: 6px; }
.conversations { flex: 1; overflow-y: auto; display: flex; flex-direction: column; gap: 4px; margin-top: 8px; }
.conv-item { text-align: left; padding: 8px 10px; background: transparent; border: none; cursor: pointer; border-radius: 6px; }
.conv-item:hover { background: #e8eaed; }
.conv-item.selected { background: #d8e3f7; }
.conv-title { font-size: 14px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.conv-date { font-size: 11px; color: #666; }
.user-footer { font-size: 12px; color: #555; padding-top: 8px; border-top: 1px solid #e1e4e8; }

.main { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
.empty-state { flex: 1; display: flex; align-items: center; justify-content: center; color: #888; }

.chat-window { display: flex; flex-direction: column; height: 100%; }
.chat-header { padding: 12px 16px; border-bottom: 1px solid #e1e4e8; font-weight: 600; }
.messages { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }

.bubble { padding: 10px 14px; border-radius: 8px; max-width: 80%; word-wrap: break-word; }
.bubble.user { background: #d8e3f7; align-self: flex-end; }
.bubble.assistant { background: #f4f5f7; align-self: flex-start; }
.bubble.editing { width: 60%; }
.bubble.editing textarea { width: 100%; min-height: 80px; }

.bubble-body p:first-child { margin-top: 0; }
.bubble-body p:last-child { margin-bottom: 0; }
.bubble-body pre { margin: 8px 0; }
.bubble-body code { background: rgba(0,0,0,0.06); padding: 1px 4px; border-radius: 3px; font-size: 90%; }
.bubble-body pre code { background: transparent; padding: 0; }

.cursor { display: inline-block; margin-left: 2px; animation: blink 1s steps(1) infinite; }
@keyframes blink { 50% { opacity: 0; } }

.sources { margin-top: 8px; font-size: 12px; color: #555; }
.sources ul { margin: 4px 0 0 20px; padding: 0; }
.stopped-marker { font-size: 12px; color: #b00; font-style: italic; margin-top: 4px; }

.bubble-actions { margin-top: 4px; display: flex; gap: 4px; }
.bubble-actions button { font-size: 11px; padding: 2px 8px; border: 1px solid #ccc; background: white; border-radius: 4px; cursor: pointer; }

.composer { display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid #e1e4e8; }
.composer textarea { flex: 1; padding: 8px; font-size: 14px; resize: none; border: 1px solid #ccc; border-radius: 6px; font-family: inherit; }
.composer button { padding: 8px 16px; font-size: 14px; cursor: pointer; background: #2d6cdf; color: white; border: none; border-radius: 6px; }
.composer button:disabled { opacity: 0.5; cursor: not-allowed; }
.composer .stop-btn { background: #c00; }

.suggested-prompts { padding: 24px; }
.suggested-prompts h3 { margin: 0 0 12px 0; color: #555; }
.prompts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.prompt-card { padding: 14px; text-align: left; background: white; border: 1px solid #ddd; border-radius: 8px; cursor: pointer; font-size: 14px; }
.prompt-card:hover { background: #f4f5f7; }

.error { padding: 8px 16px; background: #fee; color: #b00; font-size: 13px; }

.edit-actions { display: flex; gap: 8px; margin-top: 4px; }
```

- [ ] **Step 4: TypeScript check + visual test**

Run (from `frontend/`): `npx tsc --noEmit`
Expected: no errors.

Run: `npm run dev` → open `http://localhost:5173`. Expected: signin page renders.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.tsx frontend/src/App.css frontend/src/components/Sidebar.tsx
git commit -m "feat(frontend): sidebar + app shell + CSS"
```

---

## Phase 6 — End-to-end verification

### Task 18: Manual end-to-end smoke test

**Files:** none (verification only)

- [ ] **Step 1: Start RabbitMQ** (already running per the spec's pre-flight check; if not, start it).

- [ ] **Step 2: Start Python backend**

Run (from project root): `python main.py`
Expected: in logs, see `RabbitMQ consumer ready on chat.search.request` and Uvicorn on 8083.

- [ ] **Step 3: Start Java backend**

Open another terminal. Run (from `backend/`): `OPENAI_API_KEY=sk-... ./mvnw spring-boot:run`
Expected: Tomcat starts on 8080; Rabbit listener attaches to `chat.search.reply`.

- [ ] **Step 4: Start React frontend**

Open another terminal. Run (from `frontend/`): `npm run dev`
Expected: Vite serves on 5173.

- [ ] **Step 5: Walk through the user flow in the browser**

Open `http://localhost:5173`. Verify:
- Sign-in screen appears → enter "alice" → continue.
- Sidebar empty. Click "+ New Chat" → conversation appears, ChatWindow shows 4 suggested prompts.
- Click a suggested prompt → user message appears immediately, then assistant message starts streaming token-by-token with a blinking cursor; markdown renders (bold, lists); sources appear under the answer with clickable links.
- Type a follow-up question with a code block request like "Show me a Python example of risk classification". Verify code block appears with syntax highlighting.
- Hover the user message → click "✎ Edit" → modify text → Save. Verify the assistant message after it disappears and a new one streams in.
- Hover the latest assistant message → click "↻ Regenerate". Verify it re-streams.
- Submit a long question; while streaming, click "■ Stop". Verify streaming halts immediately and partial content remains visible.
- Refresh the page (F5). Verify the conversation and messages persist (H2 file).
- Open a second conversation via "+ New Chat" → confirm switching works.

- [ ] **Step 6: Commit any final tweaks**

If you made small CSS or copy fixes during the walkthrough, commit them now. Otherwise this task is verification-only.

```bash
git status
# If clean: nothing to commit. If dirty: review and commit with descriptive message.
```

---

## Notes for the executor

- **Restart Java between RabbitMQ topology changes** so bindings reapply.
- **`OPENAI_API_KEY`** must be in the Java process environment for streaming to work; the Python side already loads its own from `.env`.
- **H2 file location** (`./data/chatdb`) is relative to the Java process working directory. Run `./mvnw spring-boot:run` from `backend/` so the file lands in `backend/data/`.
- **CORS**: the React dev server proxies `/api` to Java, so CORS shouldn't fire in development. The `WebConfig` is for safety if someone hits Java directly from `localhost:5173`.
- **Edit/regenerate** rely on real DB ids. The frontend reloads messages from the server after each stream finishes (`loadMessages` in `ChatWindow.send`) so subsequent edits target the correct ids.
