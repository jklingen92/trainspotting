import os
from datetime import datetime
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = 'Record video using GStreamer with nvarguscamerasrc'

    def add_arguments(self, parser):
        parser.add_argument(
            '--sensor-id',
            type=int,
            default=0,
            help='Camera sensor ID (default: 0)'
        )
        parser.add_argument(
            '--sensor-mode',
            type=int,
            choices=[0, 1, 2],
            default=0,
            help='Sensor mode: 0 or 1 for 4K, 2 for 1080p (default: 0)'
        )
        parser.add_argument(
            '--exposure',
            type=int,
            default=450000,
            help='Exposure time in nanoseconds (default: 450000)'
        )
        parser.add_argument(
            '--bitrate',
            type=int,
            default=50000,
            help='Video bitrate in kbps (default: 50000)'
        )
        parser.add_argument(
            '--output-dir',
            type=str,
            default='./recordings',
            help='Output directory for video files (default: ./recordings)'
        )
        parser.add_argument(
            '--filename',
            type=str,
            help='Output filename (default: auto-generated with timestamp)'
        )
        parser.add_argument(
            '--duration',
            type=int,
            help='Recording duration in seconds (optional)'
        )

    def handle(self, *args, **options):
        sensor_id = options['sensor_id']
        sensor_mode = options['sensor_mode']
        exposure = options['exposure']
        bitrate = options['bitrate']
        output_dir = options['output_dir']
        filename = options['filename']
        duration = options['duration']

        # Determine resolution based on sensor mode
        if sensor_mode in [0, 1]:
            width, height = 3840, 2160  # 4K
            mode_desc = "4K"
        else:  # sensor_mode == 2
            width, height = 1920, 1080  # 1080p
            mode_desc = "1080p"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}_{mode_desc}.mp4"

        # Ensure filename has .mp4 extension
        if not filename.endswith('.mp4'):
            filename += '.mp4'

        output_path = os.path.join(output_dir, filename)

        # Build GStreamer pipeline
        pipeline_parts = [
            'gst-launch-1.0',
            f'nvarguscamerasrc sensor-id={sensor_id} sensor-mode={sensor_mode} exposuretimerange="{exposure} {exposure}"',
            '!',
            f'"video/x-raw(memory:NVMM),width={width},height={height},format=NV12,framerate=30/1"',
            '!',
            'nvvidconv',
            '!',
            'video/x-raw, format=BGRx',
            '!',
            'videoconvert',
            '!',
            'video/x-raw, format=BGR',
            '!',
            'videoconvert',
            '!',
            f'x264enc tune=zerolatency bitrate={bitrate} speed-preset=ultrafast',
            '!',
            f'video/x-h264, width={width}, height={height}, framerate={30}/1'
            '!',
            'queue',
            '!',
            'h264parse',
            '!',
            'mp4mux',
            '!',
            f'filesink location={output_path}',
            '-e'
        ]

        # Add duration if specified
        if duration:
            # Insert timeout element before filesink
            timeout_index = pipeline_parts.index('mp4mux') + 2
            pipeline_parts.insert(timeout_index, f'timeout time={duration}000000000')
            pipeline_parts.insert(timeout_index + 1, '!')

        pipeline_cmd = ' '.join(pipeline_parts)

        self.stdout.write(
            self.style.SUCCESS(
                f'Starting recording:\n'
                f'  Sensor ID: {sensor_id}\n'
                f'  Mode: {sensor_mode} ({mode_desc})\n'
                f'  Resolution: {width}x{height}\n'
                f'  Exposure: {exposure} ns\n'
                f'  Bitrate: {bitrate} kbps\n'
                f'  Output: {output_path}\n'
                f'  Duration: {"Unlimited" if not duration else f"{duration} seconds"}\n'
            )
        )

        self.stdout.write(f'Pipeline: {pipeline_cmd}\n')

        try:
            # Execute the GStreamer pipeline directly
            os.system(pipeline_cmd)
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'\nRecording completed!\n'
                    f'Output file: {output_path}'
                )
            )
                
        except KeyboardInterrupt:
            self.stdout.write(
                self.style.WARNING('\nRecording interrupted by user')
            )
        except Exception as e:
            raise CommandError(f'Unexpected error: {str(e)}')