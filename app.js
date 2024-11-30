let chatHistory = [];
let apiUrl = "";

function loadNgrokUrl() {
    return fetch('ngrokTunnel')
        .then(response => {
            if (!response.ok) throw new Error('Failed to fetch ngrok URL');
            return response.text();
        })
        .then(data => {
            const ngrokPart = data.trim();
            apiUrl = `https://${ngrokPart}.ngrok-free.app/generate`;
            console.log('API URL loaded:', apiUrl);
        })
        .catch(error => {
            console.error('Error loading ngrok URL:', error);
            alert('Error initializing chatbot. Please try again later.');
        });
}

function toggleTheme() {
    const body = document.body;
    body.classList.toggle('light-mode');
    const themeButton = document.querySelector('.theme-toggle-btn');
    themeButton.textContent = document.body.classList.contains('light-mode') ? '☀️' : '🌙';
}

function sanitizeHTML(str) {
    const temp = document.createElement('div');
    temp.textContent = str;
    return temp.innerHTML;
}

function updateChatBox() {
    const messagesDiv = document.getElementById('messages');
    messagesDiv.innerHTML = ''; 

    chatHistory.forEach(chat => {
        const userMessageDiv = document.createElement('div');
        userMessageDiv.className = 'message user-message';
        userMessageDiv.innerHTML = `<p>${sanitizeHTML(chat.user)}</p>`;
        messagesDiv.appendChild(userMessageDiv);

        const botMessageDiv = document.createElement('div');
        botMessageDiv.className = 'message bot-message';

        if (chat.bot) {
            if (chat.bot.startsWith('```') && chat.bot.endsWith('```')) {
                const codeContent = chat.bot.replace(/```/g, '').trim();
                const codeBlock = document.createElement('div');
                codeBlock.className = 'bot-code-message';
                codeBlock.innerHTML = `
                    <button class="copy-btn" onclick="copyToClipboard(this)">Copy</button>
                    <pre><code>${sanitizeHTML(codeContent)}</code></pre>
                `;
                botMessageDiv.appendChild(codeBlock);
            } else {
                botMessageDiv.innerHTML = `<p>${sanitizeHTML(chat.bot)}</p>`;
            }
        } else {
            botMessageDiv.innerHTML = `<p>No response</p>`;
        }

        messagesDiv.appendChild(botMessageDiv);
    });

    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

window.copyToClipboard = function(button) {
    const codeElement = button.nextElementSibling.querySelector('code');
    const codeText = codeElement.textContent;
    navigator.clipboard.writeText(codeText).then(() => {
        button.textContent = 'Copied!';
        setTimeout(() => {
            button.textContent = 'Copy';
        }, 2000); 
    }).catch(err => {
        console.error('Error copying code:', err);
    });
};

document.querySelector('.input-area').addEventListener('submit', function (event) {
    event.preventDefault();

    const userQuestion = document.getElementById('question').value.trim();
    if (!userQuestion) return;

    document.getElementById('question').value = '';

    chatHistory.push({ user: userQuestion, bot: 'Loading...' });
    updateChatBox();

    fetch(apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input: userQuestion }),
    })
        .then(response => {
            if (!response.ok) throw new Error('API error');
            return response.json();
        })
        .then(data => {
            let botResponse = data.answer || 'No response';

            if (botResponse.includes('{') && botResponse.includes('}') && botResponse.includes(';')) {
                botResponse = `\`\`\`${botResponse}\`\`\``;
            }

            chatHistory[chatHistory.length - 1].bot = botResponse;
            updateChatBox();
        })
        .catch(error => {
            console.error('Error:', error);
            chatHistory[chatHistory.length - 1].bot = 'Error: Unable to get a response.';
            updateChatBox();
        });
});

window.onload = function () {
    loadNgrokUrl().then(updateChatBox);
};
