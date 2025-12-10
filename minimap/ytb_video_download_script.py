import os
import cv2
import yt_dlp
from pathlib import Path

def download_youtube_video(video_url, output_video_path, resolution='1080p'):
    """
    Download YouTube video and save to local storage
    
    Parameters:
    -----------
    video_url : str
        YouTube video URL
    output_video_path : str
        Path to save the video
    resolution : str
        Video resolution (default: '1080p')
    
    Returns:
    --------
    str : Downloaded video file path
    dict : Video information
    """
    print(f"Starting YouTube video download: {video_url}")
    print(f"Target resolution: {resolution}")
    print(f"Save path: {output_video_path}")
    
    # If video already exists, return directly
    if os.path.exists(output_video_path):
        print(f"Video already exists, skipping download: {output_video_path}")
        cap = cv2.VideoCapture(output_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return output_video_path, {'width': width, 'height': height, 'fps': fps, 'duration': duration}
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_video_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    height_limit = 1080
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': f'bestvideo[height<={height_limit}][ext=mp4]+bestaudio[ext=m4a]/best[height<={height_limit}][ext=mp4]/best[height<={height_limit}]/best',
        'outtmpl': output_video_path.replace('.mp4', '.%(ext)s'),
        'merge_output_format': 'mp4',
        'quiet': False,
        'no_warnings': False,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        },
        'retries': 10,
        'fragment_retries': 10,
        'format_sort': ['res:1080', 'ext:mp4:m4a', 'hasvid', 'hasaud'],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Downloading video...")
            info = ydl.extract_info(video_url, download=True)
            
        # Find downloaded video file
        downloaded_files = []
        base_path = output_video_path.replace('.mp4', '')
        for ext in ['.mp4', '.webm', '.mkv', '.flv']:
            potential_file = base_path + ext
            if os.path.exists(potential_file):
                downloaded_files.append(potential_file)
        
        if not downloaded_files:
            # Check output directory
            output_dir = os.path.dirname(output_video_path)
            all_files = os.listdir(output_dir)
            downloaded_files = [os.path.join(output_dir, f) for f in all_files 
                               if f.endswith(('.mp4', '.webm', '.mkv', '.flv'))]
        
        if not downloaded_files:
            raise FileNotFoundError("Downloaded video file not found")
        
        video_file = downloaded_files[0]
        
        # Rename if filename is not as expected
        if video_file != output_video_path and video_file.endswith('.mp4'):
            import shutil
            shutil.move(video_file, output_video_path)
            video_file = output_video_path
        
        print(f"Video download complete: {video_file}")
        
        # Get video information
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"Video information:")
        print(f"  Resolution: {width}x{height}")
        print(f"  Frame rate: {fps:.2f} FPS")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.2f} seconds")
        
        return video_file, {'width': width, 'height': height, 'fps': fps, 'duration': duration}
        
    except Exception as e:
        print(f"Download failed: {e}")
        # Try simplified configuration
        print("Trying simplified configuration...")
        ydl_opts_simple = {
            'format': 'best[height<=1080]/best',
            'outtmpl': output_video_path.replace('.mp4', '.%(ext)s'),
            'merge_output_format': 'mp4',
            'quiet': False,
            'no_warnings': False,
            'user_agent': 'Mozilla/5.0',
            'retries': 5,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts_simple) as ydl:
                info = ydl.extract_info(video_url, download=True)
            
            # Find downloaded file
            base_path = output_video_path.replace('.mp4', '')
            for ext in ['.mp4', '.webm', '.mkv']:
                potential_file = base_path + ext
                if os.path.exists(potential_file):
                    if potential_file != output_video_path:
                        import shutil
                        shutil.move(potential_file, output_video_path)
                    
                    # Get video information
                    cap = cv2.VideoCapture(output_video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    
                    return output_video_path, {'width': width, 'height': height, 'fps': fps, 'duration': 0}
        except Exception as e2:
            raise Exception(f"Download failed. Please try:\n1. Update yt-dlp: pip install -U yt-dlp\n2. Check network connection\n3. Video may be unavailable\nError: {e2}")


def extract_clip(input_video_path, output_clip_path, num_frames=2000):
    """
    Extract first N frames from video and save as a clip
    
    Parameters:
    -----------
    input_video_path : str
        Input video file path
    output_clip_path : str
        Output clip file path
    num_frames : int
        Number of frames to extract (default: 2000)
    
    Returns:
    --------
    dict : Clip information
    """
    print(f"\n{'='*80}")
    print(f"Extracting first {num_frames} frames from video...")
    print(f"{'='*80}")
    
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        raise Exception(f"Failed to open video: {input_video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Adjust num_frames if video is shorter
    actual_frames = min(num_frames, total_frames)
    
    print(f"Input video info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Frames to extract: {actual_frames}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_clip_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        raise Exception(f"Failed to create output video: {output_clip_path}")
    
    # Extract frames
    frame_count = 0
    while frame_count < actual_frames:
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could only extract {frame_count} frames")
            break
        
        out.write(frame)
        frame_count += 1
        
        # Progress indicator
        if frame_count % 100 == 0:
            progress = (frame_count / actual_frames) * 100
            print(f"Progress: {frame_count}/{actual_frames} frames ({progress:.1f}%)")
    
    # Release resources
    cap.release()
    out.release()
    
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Clip extraction complete!")
    print(f"{'='*80}")
    print(f"Extracted frames: {frame_count}")
    print(f"Clip duration: {duration:.2f} seconds")
    print(f"Saved to: {output_clip_path}")
    
    return {
        'frames': frame_count,
        'duration': duration,
        'width': width,
        'height': height,
        'fps': fps
    }