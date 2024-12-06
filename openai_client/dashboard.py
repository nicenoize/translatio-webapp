# dashboard.py

from aiohttp import web
import asyncio
import logging
import time  # Added import for time module


class Dashboard:
    def __init__(self, client, logger: logging.Logger):
        self.client = client
        self.logger = logger
        self.dashboard_task = asyncio.create_task(self.run_dashboard_server())

    async def run_dashboard_server(self):
        """Run the real-time monitoring dashboard using aiohttp."""
        app = web.Application()
        app.add_routes([
            web.get('/', self.handle_dashboard),
            web.get('/metrics', self.metrics_endpoint)
        ])

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.client.DASHBOARD_PORT)
        await site.start()
        self.logger.info(f"Monitoring dashboard started at http://localhost:{self.client.DASHBOARD_PORT}")

        # Keep the dashboard running
        while self.client.running:
            await asyncio.sleep(3600)

    async def handle_dashboard(self, request):
        """Handle HTTP requests to the dashboard."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Real-Time Translator Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; }}
                .chart-container {{
                    width: 100%;
                    height: 400px;
                    margin-top: 20px;
                }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script>
                async function fetchMetrics() {{
                    const response = await fetch('/metrics');
                    const data = await response.json();
                    document.getElementById('avg_delay').innerText = data.average_processing_delay.toFixed(6);
                    document.getElementById('min_delay').innerText = data.min_processing_delay.toFixed(6);
                    document.getElementById('max_delay').innerText = data.max_processing_delay.toFixed(6);
                    document.getElementById('stddev_delay').innerText = data.stddev_processing_delay.toFixed(6);
                    document.getElementById('buffer_status').innerText = data.buffer_status;
                    document.getElementById('audio_queue_size').innerText = data.audio_queue_size;
                    document.getElementById('muxing_queue_size').innerText = data.muxing_queue_size;

                    // New RTMPStreamer metrics
                    document.getElementById('total_segments').innerText = data.total_segments_streamed;
                    document.getElementById('ffmpeg_errors').innerText = data.ffmpeg_errors;
                    document.getElementById('ffmpeg_uptime').innerText = (data.ffmpeg_uptime).toFixed(2) + 's';

                    // Update chart data
                    if (window.processingDelayChart) {{
                        window.processingDelayChart.data.labels = data.processing_delays.map((_, i) => i + 1);
                        window.processingDelayChart.data.datasets[0].data = data.processing_delays;
                        window.processingDelayChart.update();
                    }}
                }}

                window.onload = function() {{
                    const ctx = document.getElementById('processingDelayChart').getContext('2d');
                    window.processingDelayChart = new Chart(ctx, {{
                        type: 'line',
                        data: {{
                            labels: [],
                            datasets: [{{
                                label: 'Processing Delay (s)',
                                data: [],
                                borderColor: 'rgba(75, 192, 192, 1)',
                                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                                fill: true,
                                tension: 0.1
                            }}]
                        }},
                        options: {{
                            scales: {{
                                x: {{
                                    title: {{
                                        display: true,
                                        text: 'Sample'
                                    }}
                                }},
                                y: {{
                                    beginAtZero: true,
                                    title: {{
                                        display: true,
                                        text: 'Delay (s)'
                                    }}
                                }}
                            }}
                        }}
                    }});

                    fetchMetrics();  // Initial fetch
                    setInterval(fetchMetrics, 1000);
                }};
            </script>
        </head>
        <body>
            <h1>Real-Time Translator Dashboard</h1>
            <h2>Streaming Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Average Processing Delay (s)</td><td id="avg_delay">0.0</td></tr>
                <tr><td>Minimum Processing Delay (s)</td><td id="min_delay">0.0</td></tr>
                <tr><td>Maximum Processing Delay (s)</td><td id="max_delay">0.0</td></tr>
                <tr><td>Std Dev Processing Delay (s)</td><td id="stddev_delay">0.0</td></tr>
                <tr><td>Audio Buffer Size</td><td id="buffer_status">0</td></tr>
                <tr><td>Translated Audio Queue Size</td><td id="audio_queue_size">0</td></tr>
                <tr><td>Muxing Queue Size</td><td id="muxing_queue_size">0</td></tr>
                <!-- New Metrics -->
                <tr><td>Total Segments Streamed</td><td id="total_segments">0</td></tr>
                <tr><td>FFmpeg Errors</td><td id="ffmpeg_errors">0</td></tr>
                <tr><td>FFmpeg Uptime</td><td id="ffmpeg_uptime">0.00s</td></tr>
            </table>

            <div class="chart-container">
                <canvas id="processingDelayChart"></canvas>
            </div>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    async def metrics_endpoint(self, request):
        """Provide metrics in JSON format for the dashboard."""
        # Convert deque to list for JSON serialization
        serializable_metrics = self.client.metrics.copy()
        serializable_metrics["processing_delays"] = list(serializable_metrics["processing_delays"])

        # Include RTMPStreamer metrics
        rtmp_metrics = {
            "total_segments_streamed": getattr(self.client.rtmp_streamer, 'total_segments_streamed', 0),
            "ffmpeg_errors": getattr(self.client.rtmp_streamer, 'ffmpeg_errors', 0),
            "ffmpeg_uptime": (time.time() - getattr(self.client.rtmp_streamer, 'ffmpeg_start_time', time.time())) 
                                if getattr(self.client.rtmp_streamer, 'ffmpeg_start_time', None) else 0.0
        }
        serializable_metrics.update(rtmp_metrics)

        return web.json_response(serializable_metrics)
