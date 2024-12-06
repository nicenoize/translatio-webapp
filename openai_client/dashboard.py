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
                body {{
                    font-family: 'Arial', sans-serif;
                    background-color: #121212;
                    color: #e0e0e0;
                    margin: 0;
                    padding: 0;
                }}
                h1, h2 {{
                    color: #EC7D0D;
                    text-align: center;
                }}
                .dashboard-container {{
                    padding: 20px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border: 1px solid #444;
                    color: #e0e0e0;
                }}
                th {{
                    background-color: #222;
                }}
                td {{
                    background-color: #1e1e1e;
                }}
                .chart-container {{
                    background-color: #1e1e1e;
                    border: 1px solid #444;
                    border-radius: 8px;
                    padding: 20px;
                    margin-top: 20px;
                    /* Increased width and height */
                    width: 90%; /* Increased from 80% to 90% */
                    max-width: 1200px; /* Increased from 1000px to 1200px */
                    height: 600px; /* Set a specific height */
                    margin-left: auto;
                    margin-right: auto;
                }}
                canvas {{
                    display: block;
                    margin: 0 auto;
                    width: 100% !important; /* Ensure the canvas takes full width of the container */
                    height: 100% !important; /* Ensure the canvas takes full height of the container */
                }}
                .metric-value {{
                    font-weight: bold;
                    color: #f1c40f;
                }}
                .button {{
                    display: inline-block;
                    padding: 10px 20px;
                    font-size: 16px;
                    color: #fff;
                    background-color: #f39c12;
                    border: none;
                    border-radius: 5px;
                    text-decoration: none;
                    text-align: center;
                    cursor: pointer;
                }}
                .button:hover {{
                    background-color: #e67e22;
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
                                borderColor: '#EC7D0D', /* Changed from blue to orange */
                                backgroundColor: 'rgba(236, 125, 13, 0.2)', /* Changed to semi-transparent orange */
                                pointBackgroundColor: '#EC7D0D', /* Optional: Set point color to orange */
                                pointBorderColor: '#EC7D0D', /* Optional: Set point border to orange */
                                pointHoverBackgroundColor: '#fff', /* Optional: Hover effect */
                                pointHoverBorderColor: '#EC7D0D', /* Optional: Hover effect */
                                fill: true,
                                tension: 0.1
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false, /* Allow the chart to adjust to the container size */
                            scales: {{
                                x: {{
                                    title: {{
                                        display: true,
                                        text: 'Sample',
                                        color: '#EC7D0D' /* Optional: Title color */
                                    }},
                                    ticks: {{
                                        color: '#EC7D0D' /* Set X-axis tick color to orange */
                                    }},
                                    grid: {{
                                        color: 'rgba(236, 125, 13, 0.1)' /* Optional: Grid lines color */
                                    }}
                                }},
                                y: {{
                                    beginAtZero: true,
                                    title: {{
                                        display: true,
                                        text: 'Delay (s)',
                                        color: '#EC7D0D' /* Optional: Title color */
                                    }},
                                    ticks: {{
                                        color: '#EC7D0D' /* Set Y-axis tick color to orange */
                                    }},
                                    grid: {{
                                        color: 'rgba(236, 125, 13, 0.1)' /* Optional: Grid lines color */
                                    }}
                                }}
                            }},
                            plugins: {{
                                legend: {{
                                    labels: {{
                                        color: '#EC7D0D' /* Set legend text color to orange */
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
