from youtube_transcript_api import YouTubeTranscriptApi
import re

def extract_youtube_transcript(video_url):
    """
    Extract transcript from a YouTube video URL
    
    Args:
        video_url (str): YouTube video URL
        
    Returns:
        str: Formatted transcript text
    """
    # Extract video ID from URL
    video_id = extract_video_id(video_url)
    
    if not video_id:
        return "Invalid YouTube URL"
    
    try:
        # Get transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Format transcript
        formatted_transcript = ""
        for entry in transcript:
            formatted_transcript += f"[{entry['start']:.1f}s] {entry['text']}\n"
        
        return formatted_transcript
    
    except Exception as e:
        return f"Error extracting transcript: {str(e)}"

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:v\/|embed\/|watch\?v=|watch\?.+&v=)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def save_transcript(transcript, filename):
    """Save transcript to a text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(transcript)
    print(f"Transcript saved to {filename}")

# Example usage
if __name__ == "__main__":
    # Replace with your YouTube video URL
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    print("Extracting transcript...")
    transcript = extract_youtube_transcript(video_url)
    
    print("\nTranscript:")
    print(transcript)
    
    # Optionally save to file
    # save_transcript(transcript, "transcript.txt")