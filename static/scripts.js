// static/script.js

document.addEventListener('DOMContentLoaded', () => {
    const subtitles = document.getElementById('subtitles');
    const audioPlayer = document.getElementById('translatedAudio');

    // Determine the WebSocket protocol based on the page's protocol
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    let socketUrl = `${protocol}://${window.location.host}/ws/translations`;
    let socket = new WebSocket(socketUrl);

    // Audio Queue to handle sequential playback
    const audioQueue = [];
    let isPlaying = false;

    // Fixed delay in milliseconds to synchronize audio with video
    const fixedDelay = 2000; // 2 seconds

    // Timestamp when the audio should start playing
    let scheduledStartTime = null;

    // Keep track of created blob URLs for cleanup
    const blobUrls = new Set();

    socket.onopen = () => {
        console.log('WebSocket connection established');
        clearSubtitles();
        displaySubtitles('Connected to server. Waiting for translation...');
    };

    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            console.log('Received message:', data.type);

            if (data.type === 'translation') {
                displaySubtitles(data.text);
            } else if (data.type === 'audio') {
                enqueueAudio(data.audio);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };

    socket.onclose = (event) => {
        console.log('WebSocket connection closed:', event);
        displaySubtitles('Connection closed. Attempting to reconnect...');
        attemptReconnect();
    };

    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        displaySubtitles('WebSocket error. Check console for details.');
    };

    function clearSubtitles() {
        subtitles.innerHTML = '';
    }

    function displaySubtitles(text) {
        // Remove "waiting" message if present
        if (subtitles.querySelector('.waiting-message')) {
            clearSubtitles();
        }

        const newSubtitle = document.createElement('div');
        newSubtitle.textContent = text;
        newSubtitle.className = 'mb-2 p-2 bg-gray-800 bg-opacity-75 rounded text-white inline-block';
        
        // Add timestamp
        const timestamp = new Date().toLocaleTimeString();
        const timeSpan = document.createElement('span');
        timeSpan.textContent = `[${timestamp}] `;
        timeSpan.className = 'text-gray-400 text-sm';
        newSubtitle.insertBefore(timeSpan, newSubtitle.firstChild);

        subtitles.appendChild(newSubtitle);

        // Keep only last 10 subtitles to prevent memory issues
        while (subtitles.children.length > 10) {
            subtitles.removeChild(subtitles.firstChild);
        }

        // Auto-scroll to the latest subtitle
        subtitles.scrollTop = subtitles.scrollHeight;
    }

    function enqueueAudio(base64Audio) {
        audioQueue.push(base64Audio);
        processAudioQueue();
    }

    function cleanupBlobUrl(url) {
        if (blobUrls.has(url)) {
            URL.revokeObjectURL(url);
            blobUrls.delete(url);
        }
    }

    function processAudioQueue() {
        if (isPlaying || audioQueue.length === 0) return;

        isPlaying = true;
        const base64Audio = audioQueue.shift();

        try {
            // Decode the base64 audio string to binary data
            const binaryString = window.atob(base64Audio);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }

            // Create a Blob from the binary data
            const blob = new Blob([bytes.buffer], { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(blob);
            blobUrls.add(audioUrl);

            // Clean up previous audio URL
            if (audioPlayer.src) {
                cleanupBlobUrl(audioPlayer.src);
            }

            // Set the source of the audio player
            audioPlayer.src = audioUrl;

            // Calculate the scheduled start time
            if (!scheduledStartTime) {
                scheduledStartTime = Date.now() + fixedDelay;
            }

            const delay = scheduledStartTime - Date.now();

            // Play the audio after the calculated delay
            setTimeout(() => {
                audioPlayer.play().then(() => {
                    console.log('Playing translated audio...');
                }).catch(error => {
                    console.error('Error playing translated audio:', error);
                    isPlaying = false;
                    cleanupBlobUrl(audioUrl);
                    processAudioQueue(); // Attempt to play the next audio
                });
            }, Math.max(delay, 0));

            // Listen for the end of the audio playback
            audioPlayer.onended = () => {
                isPlaying = false;
                cleanupBlobUrl(audioUrl);
                scheduledStartTime = Date.now() + fixedDelay;
                processAudioQueue();
            };

            // Update the scheduled start time for the next audio
            scheduledStartTime += fixedDelay;

        } catch (error) {
            console.error('Error processing audio:', error);
            isPlaying = false;
            processAudioQueue();
        }
    }

    async function attemptReconnect() {
        const retryInterval = 5000; // 5 seconds
        
        // Clean up existing connection
        if (socket) {
            socket.close();
        }

        await new Promise(resolve => setTimeout(resolve, retryInterval));

        console.log('Attempting to reconnect WebSocket...');
        displaySubtitles('Reconnecting...');

        // Re-initialize the WebSocket connection
        socket = new WebSocket(socketUrl);

        // Reattach all event handlers
        socket.onopen = () => {
            console.log('WebSocket reconnected');
            clearSubtitles();
            displaySubtitles('Reconnected to server. Waiting for translation...');
        };

        socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === 'translation') {
                    displaySubtitles(data.text);
                } else if (data.type === 'audio') {
                    enqueueAudio(data.audio);
                }
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };

        socket.onclose = (event) => {
            console.log('WebSocket connection closed:', event);
            displaySubtitles('Connection closed. Attempting to reconnect...');
            attemptReconnect();
        };

        socket.onerror = (error) => {
            console.error('WebSocket error:', error);
            displaySubtitles('WebSocket error. Check console for details.');
        };
    }

    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        // Clean up all blob URLs
        blobUrls.forEach(url => {
            cleanupBlobUrl(url);
        });
        
        // Close WebSocket connection
        if (socket) {
            socket.close();
        }
    });
});