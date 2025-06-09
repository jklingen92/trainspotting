import os
import glob
from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from detection.motion_capture import check_for_train, check_for_train_with_diagnostics


class Command(BaseCommand):
    help = 'Analyze video files for train detection and optionally delete non-train videos'

    def add_arguments(self, parser):
        parser.add_argument(
            'path',
            type=str,
            help='Path to video files (can be a directory or glob pattern like "/path/to/videos/*.mp4")'
        )
        
        parser.add_argument(
            '--threshold',
            type=int,
            default=10,
            help='Motion detection threshold (default: 10)'
        )
        
        parser.add_argument(
            '--no-cuda',
            action='store_true',
            help='Disable CUDA acceleration'
        )
        
        parser.add_argument(
            '--delete',
            action='store_true',
            help='Delete video files that do not contain trains (False results)'
        )
        
        parser.add_argument(
            '--extensions',
            type=str,
            default='mp4,avi,mov,mkv,webm',
            help='Comma-separated list of video file extensions to process (default: mp4,avi,mov,mkv,webm)'
        )
        
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output with diagnostic information'
        )

    def handle(self, *args, **options):
        path = options['path']
        threshold = options['threshold']
        use_cuda = not options['no_cuda']
        delete = options['delete']
        verbose = options['verbose']
        extensions = [ext.strip().lower() for ext in options['extensions'].split(',')]

        # Get list of video files
        video_files = self.get_video_files(path, extensions)
        
        if not video_files:
            self.stdout.write(
                self.style.WARNING(f'No video files found in: {path}')
            )
            return

        self.stdout.write(f'Found {len(video_files)} video files to process...\n')

        # Statistics
        stats = {
            'total': len(video_files),
            'trains': 0,
            'no_trains': 0,
            'anomalous': 0,
            'errors': 0,
            'deleted': 0
        }

        files_to_delete = []

        # Process each video file
        for i, video_file in enumerate(video_files, 1):
            self.stdout.write(f'[{i}/{len(video_files)}] Processing: {os.path.basename(video_file)}')
            
            try:
                if verbose:
                    result, diagnostics = check_for_train_with_diagnostics(
                        video_file, threshold=threshold, use_cuda=use_cuda
                    )
                    if 'error' in diagnostics:
                        self.stdout.write(
                            self.style.ERROR(f'  ERROR: {diagnostics["error"]}')
                        )
                        stats['errors'] += 1
                        continue
                else:
                    result = check_for_train(video_file, threshold=threshold, use_cuda=use_cuda)
                    diagnostics = None

                # Display result
                if result is True:
                    self.stdout.write(
                        self.style.SUCCESS('  RESULT: Train detected ✓')
                    )
                    stats['trains'] += 1
                elif result is False:
                    self.stdout.write(
                        self.style.WARNING('  RESULT: No train detected')
                    )
                    stats['no_trains'] += 1
                    if delete:
                        files_to_delete.append(video_file)
                elif result is None:
                    self.stdout.write(
                        self.style.ERROR('  RESULT: Anomalous pattern detected')
                    )
                    stats['anomalous'] += 1

                # Show diagnostic info if verbose
                if verbose and diagnostics and 'error' not in diagnostics:
                    motion = diagnostics['motion_values']
                    self.stdout.write(
                        f'    Motion: start→mid={motion["start_to_middle"]:.1f}%, '
                        f'mid→end={motion["middle_to_end"]:.1f}%, '
                        f'start→end={motion["start_to_end"]:.1f}%'
                    )
                    self.stdout.write(f'    Analysis: {diagnostics["pattern_analysis"]}')
                    
                    video_info = diagnostics['video_info']
                    self.stdout.write(
                        f'    Video: {video_info["n_frames_used"]} frames, '
                        f'{video_info["fps"]:.1f} fps, '
                        f'{video_info["width"]}x{video_info["height"]}'
                    )

            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f'  ERROR: {str(e)}')
                )
                stats['errors'] += 1

            self.stdout.write('')  # Empty line for readability

        # Handle file deletion
        if files_to_delete:
            self.stdout.write(f'\n{len(files_to_delete)} files marked for deletion:')
            for file_path in files_to_delete:
                self.stdout.write(f'  - {os.path.basename(file_path)}')

            self.stdout.write('\nDeleting files...')
            for file_path in files_to_delete:
                try:
                    os.remove(file_path)
                    self.stdout.write(
                        self.style.SUCCESS(f'  Deleted: {os.path.basename(file_path)}')
                    )
                    stats['deleted'] += 1
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f'  Failed to delete {os.path.basename(file_path)}: {str(e)}')
                    )

        # Print summary
        self.stdout.write('\n' + '='*50)
        self.stdout.write('SUMMARY:')
        self.stdout.write(f'  Total files processed: {stats["total"]}')
        self.stdout.write(
            self.style.SUCCESS(f'  Trains detected: {stats["trains"]}')
        )
        self.stdout.write(
            self.style.WARNING(f'  No trains: {stats["no_trains"]}')
        )
        self.stdout.write(
            self.style.ERROR(f'  Anomalous results: {stats["anomalous"]}')
        )
        if stats['errors'] > 0:
            self.stdout.write(
                self.style.ERROR(f'  Processing errors: {stats["errors"]}')
            )
        if stats['deleted'] > 0:
            self.stdout.write(
                self.style.SUCCESS(f'  Files deleted: {stats["deleted"]}')
            )

    def get_video_files(self, path, extensions):
        """Get list of video files from path or glob pattern"""
        video_files = []
        
        # If path contains wildcards, treat as glob pattern
        if '*' in path or '?' in path:
            video_files = glob.glob(path)
        # If path is a directory, find all video files
        elif os.path.isdir(path):
            for ext in extensions:
                pattern = os.path.join(path, f'*.{ext}')
                video_files.extend(glob.glob(pattern))
                # Also check uppercase extensions
                pattern = os.path.join(path, f'*.{ext.upper()}')
                video_files.extend(glob.glob(pattern))
        # If path is a single file
        elif os.path.isfile(path):
            video_files = [path]
        
        # Filter by extensions if files were found via glob
        if video_files:
            filtered_files = []
            for file_path in video_files:
                file_ext = os.path.splitext(file_path)[1][1:].lower()  # Remove dot and lowercase
                if file_ext in extensions:
                    filtered_files.append(file_path)
            video_files = filtered_files
        
        # Sort files for consistent processing order
        return sorted(video_files)