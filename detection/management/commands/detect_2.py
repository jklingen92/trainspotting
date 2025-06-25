#!/usr/bin/env python3

from django.core.management.base import BaseCommand, CommandError
import os
import logging

from detection.train_detection import CUDA_AVAILABLE, TrainDetectionSystem

logger = logging.getLogger('detection')


class Command(BaseCommand):
    help = 'Run the train motion detection system with CSI camera'

    def add_arguments(self, parser):
        # CSI Camera settings
        parser.add_argument(
            '--sensor-mode',
            type=int,
            choices=[0, 1, 2],
            default=2,
            help='CSI camera sensor mode: 0,1=4K (3840x2160), 2=1080p (1920x1080) (default: 2)'
        )
        parser.add_argument(
            '--exposure',
            type=int,
            default=450000,
            help='Camera exposure time in nanoseconds (default: 450000)'
        )
        parser.add_argument(
            '--fps',
            type=int,
            default=30,
            help='Camera frame rate (default: 30)'
        )
        parser.add_argument(
            '--video-quality',
            type=str,
            choices=['low', 'medium', 'high'],
            default='medium',
            help='Video quality setting: low=10Mbps, medium=20Mbps, high=50Mbps (default: medium)'
        )
        
        # Recording settings
        parser.add_argument(
            '--output-dir',
            type=str,
            default='./recordings',
            help='Output directory for recordings (default: ./recordings)'
        )
        parser.add_argument(
            '--buffer-seconds',
            type=int,
            default=1,
            help='Buffer seconds before/after motion (default: 1)'
        )
        parser.add_argument(
            '--min-recording-duration',
            type=float,
            default=3.0,
            help='Minimum recording duration in seconds (default: 3.0)'
        )
        
        # Motion detection settings
        parser.add_argument(
            '--motion-threshold',
            type=int,
            default=5000,
            help='Motion detection threshold (default: 5000)'
        )
        
        # Analysis settings
        parser.add_argument(
            '--disable-car-counting',
            action='store_true',
            help='Disable train car counting'
        )
        parser.add_argument(
            '--disable-speed-calculation',
            action='store_true',
            help='Disable speed calculation'
        )
        parser.add_argument(
            '--disable-placard-detection',
            action='store_true',
            help='Disable placard detection'
        )
        parser.add_argument(
            '--speed-calibration-factor',
            type=float,
            default=0.1,
            help='Speed calibration factor (pixels/second to mph) (default: 0.1)'
        )
        
        # SSH upload settings
        parser.add_argument(
            '--remote-host',
            type=str,
            help='Remote SSH host for uploads'
        )
        parser.add_argument(
            '--remote-user',
            type=str,
            help='Remote SSH username'
        )
        parser.add_argument(
            '--remote-path',
            type=str,
            help='Remote path for uploads'
        )
        parser.add_argument(
            '--ssh-key',
            type=str,
            help='SSH private key file path'
        )
        
        # CUDA settings
        parser.add_argument(
            '--disable-cuda',
            action='store_true',
            help='Disable CUDA acceleration (use CPU only)'
        )
        parser.add_argument(
            '--gpu-memory-fraction',
            type=float,
            default=0.8,
            help='GPU memory fraction to use (default: 0.8)'
        )
        
        # Logging settings
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose debug logging'
        )

    def handle(self, *args, **options):
        """Main command handler"""
        try:
            # Validate output directory
            output_dir = options['output_dir']
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    logger.info(f"Created output directory: {output_dir}")
                except Exception as e:
                    raise CommandError(f"Cannot create output directory {output_dir}: {e}")
            
            # Validate SSH configuration
            if options['remote_host'] and not options['remote_user']:
                raise CommandError("Remote user must be specified when using remote host")
            
            if options['ssh_key'] and not os.path.exists(options['ssh_key']):
                raise CommandError(f"SSH key file not found: {options['ssh_key']}")
            
            # Validate exposure range (typical range for IMX219 sensor)
            exposure = options['exposure']
            if exposure < 13000 or exposure > 683709000:
                logger.warning(f"Exposure {exposure} may be outside valid range (13000-683709000 ns)")
            
            # Set log level based on verbose flag
            if options['verbose']:
                logging.getLogger('detection').setLevel(logging.DEBUG)
                logger.info("Verbose logging enabled")
            
            # Convert disable flags to enable flags
            options['enable_car_counting'] = not options.pop('disable_car_counting', False)
            options['enable_speed_calculation'] = not options.pop('disable_speed_calculation', False)
            options['enable_placard_detection'] = not options.pop('disable_placard_detection', False)
            
            # Handle CUDA setting
            options['use_cuda'] = not options.pop('disable_cuda', False) and CUDA_AVAILABLE
            
            if options['use_cuda']:
                logger.info("CUDA acceleration enabled")
            else:
                logger.info("Using CPU-only processing")
            
            # Log camera configuration
            sensor_mode = options['sensor_mode']
            exposure = options['exposure']
            if sensor_mode in [0, 1]:
                resolution = "4K (3840x2160)"
            else:
                resolution = "1080p (1920x1080)"
            
            logger.info(f"CSI Camera Configuration:")
            logger.info(f"  Sensor Mode: {sensor_mode}")
            logger.info(f"  Resolution: {resolution}")
            logger.info(f"  Exposure: {exposure} ns")
            logger.info(f"  Frame Rate: {options['fps']} fps")
            logger.info(f"  Video Quality: {options['video_quality']}")
            
            # Create and run the detection system
            self.stdout.write(self.style.SUCCESS('Starting train detection system...'))
            detector = TrainDetectionSystem(**options)
            detector.run()
            
        except KeyboardInterrupt:
            self.stdout.write(self.style.SUCCESS('Detection system stopped by user'))
        except Exception as e:
            logger.error(f"Error running detection system: {e}")
            raise CommandError(f"Error running detection system: {e}")
        finally:
            self.stdout.write(self.style.SUCCESS('Train detection system shutdown complete'))