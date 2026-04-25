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
