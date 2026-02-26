import os
import json
import glob

folder_path = "song_ci"
file_pattern = os.path.join(folder_path, "ci.song.*.json")
files = glob.glob(file_pattern)

seen_poems = {}
duplicates = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
        for poem in data:
            # Join paragraphs into one string
            poem_text = "".join(poem["paragraphs"]).strip()
            
            if poem_text in seen_poems:
                duplicates.append((poem, seen_poems[poem_text]))
            else:
                seen_poems[poem_text] = poem

print(f"Total poems checked: {sum(len(json.load(open(f, encoding='utf-8'))) for f in files)}")
print(f"Number of duplicate poems found: {len(duplicates)}")
