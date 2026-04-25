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
