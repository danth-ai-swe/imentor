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
