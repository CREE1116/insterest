import DevConsole from './components/DevConsole';

function App() {
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <header style={{ backgroundColor: '#0f172a', padding: '1rem 2rem', borderBottom: '1px solid #1e293b', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}>
        <div style={{ maxWidth: '1400px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span style={{ fontSize: '1.75rem' }}>📊</span>
            <div>
              <h1 style={{ color: 'white', fontSize: '1.25rem', fontWeight: 900, letterSpacing: '-0.04em', lineHeight: '1.1' }}>Interest Engine Console</h1>
              <span style={{ color: '#38bdf8', fontSize: '0.6875rem', fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase' }}>Diagnostics & Benchmarking Portal</span>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#22c55e', boxShadow: '0 0 10px #22c55e' }} />
            <span style={{ color: '#94a3b8', fontSize: '0.75rem', fontWeight: 800 }}>Vectordb-HNSW-v2 Connected</span>
          </div>
        </div>
      </header>
      <main style={{ flex: 1, paddingBottom: '4rem' }}>
        <DevConsole />
      </main>
      <footer style={{ borderTop: '1px solid #e2e8f0', padding: '1.5rem 2rem', textAlign: 'center', color: '#94a3b8', fontSize: '0.8125rem', fontWeight: 600 }}>
        © {new Date().getFullYear()} Interest AI Group. Powered by CLIP & Redis HNSW Indexing.
      </footer>
    </div>
  );
}

export default App;
