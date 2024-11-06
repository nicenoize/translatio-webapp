// static/script.js

document.addEventListener('DOMContentLoaded', () => {
    const translationDisplay = document.getElementById('translationDisplay');
    const audioPlayer = document.getElementById('translatedAudio');

    // Determine the WebSocket protocol based on the page's protocol
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const socketUrl = `${protocol}://${window.location.host}/ws/translations`;
    const socket = new WebSocket(socketUrl);

    // Audio Queue to handle sequential playback
    const audioQueue = [];
    let isPlaying = false;

    socket.onopen = () => {
        console.log('WebSocket connection established');
    };

    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.type === 'translation') {
                displayTranslation(data.text);
            } else if (data.type === 'audio') {
                enqueueAudio(data.audio);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };

    socket.onclose = (event) => {
        console.log('WebSocket connection closed:', event);
        displayConnectionStatus('Connection closed. Attempting to reconnect...');
        attemptReconnect();
    };

    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        displayConnectionStatus('WebSocket error. Check console for details.');
    };

    function displayTranslation(text) {
        // If it's the first translation, clear the placeholder text
        if (translationDisplay.textContent.trim() === 'Waiting for translation...') {
            translationDisplay.textContent = '';
        }

        const newTranslation = document.createElement('div');
        newTranslation.textContent = text;
        newTranslation.className = 'mb-2 p-2 bg-white rounded';
        translationDisplay.appendChild(newTranslation);

        // Auto-scroll to the latest translation
        translationDisplay.scrollTop = translationDisplay.scrollHeight;
    }

    function enqueueAudio(base64Audio) {
        audioQueue.push(base64Audio);
        processAudioQueue();
    }

    function processAudioQueue() {
        if (isPlaying || audioQueue.length === 0) return;

        isPlaying = true;
        const base64Audio = audioQueue.shift();

        // Decode the base64 audio string to binary data
        const binaryString = window.atob(base64Audio);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }

        // Create a Blob from the binary data
        const blob = new Blob([bytes.buffer], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(blob);

        // Set the source of the audio player
        audioPlayer.src = audioUrl;

        // Play the audio
        audioPlayer.play().then(() => {
            console.log('Playing audio...');
        }).catch(error => {
            console.error('Error playing audio:', error);
            isPlaying = false;
            processAudioQueue(); // Attempt to play the next audio
        });

        // Listen for the end of the audio playback
        audioPlayer.onended = () => {
            isPlaying = false;
            processAudioQueue();
        };
    }

    function displayConnectionStatus(status) {
        const statusDiv = document.createElement('div');
        statusDiv.textContent = status;
        statusDiv.className = 'mb-2 p-2 bg-red-100 text-red-700 rounded';
        translationDisplay.appendChild(statusDiv);
        translationDisplay.scrollTop = translationDisplay.scrollHeight;
    }

    function attemptReconnect() {
        const retryInterval = 5000; // 5 seconds
        setTimeout(() => {
            console.log('Attempting to reconnect WebSocket...');
            // Re-initialize the WebSocket connection
            const newSocket = new WebSocket(`${protocol}://${window.location.host}/ws/translations`);

            newSocket.onopen = () => {
                console.log('WebSocket reconnected');
                displayConnectionStatus('Reconnected to server.');
                // Replace the old socket with the new one
                socket = newSocket;
            };

            newSocket.onmessage = socket.onmessage;
            newSocket.onclose = socket.onclose;
            newSocket.onerror = socket.onerror;
        }, retryInterval);
    }
});
