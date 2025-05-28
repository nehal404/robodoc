const fetch = require('node-fetch');

exports.handler = async (event) => {
    if (event.httpMethod !== 'POST') {
        return { statusCode: 405, body: 'Method Not Allowed' };
    }

    const { message } = JSON.parse(event.body);
    const apiKey = process.env.GROQ_API_KEY; // Stored securely in Netlify

    try {
        const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${apiKey}`,
                'Content-Type': 'application/json; charset=UTF-8',
            },
            body: JSON.stringify({
                messages: [{ role: 'user', content: message }],
                model: 'meta-llama/llama-4-maverick-17b-128e-instruct',
                temperature: 0.5,
                max_tokens: 1024,
            }),
        });

        const data = await response.json();
        return {
            statusCode: 200,
            body: JSON.stringify({ response: data.choices[0].message.content }),
        };
    } catch (error) {
        return {
            statusCode: 500,
            body: JSON.stringify({ error: 'Failed to process request' }),
        };
    }
};