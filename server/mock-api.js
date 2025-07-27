app.post('/phq9-chat', async (req, res) => {
  const { messages, session_id } = req.body;

  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: 'Messages must be a non-empty array' });
  }

  try {
    const fastapiRes = await fetch('http://localhost:8000/phq9-chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages, session_id }) // ✅ Forward session_id
    });

    if (!fastapiRes.ok) {
      const errorText = await fastapiRes.text();
      return res.status(500).json({ error: 'Model error', details: errorText });
    }

    const data = await fastapiRes.json();
    res.json({ response: data.response });
  } catch (err) {
    console.error('Error forwarding to model:', err);
    res.status(500).json({ error: 'Could not connect to backend model' });
  }
});
