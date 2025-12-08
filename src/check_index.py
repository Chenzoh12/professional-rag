import chromadb

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("professional_docs")

print(f"Total documents in index: {collection.count()}")
print("\n" + "="*60)

# Get ALL documents
results = collection.get(include=['metadatas'])

print(f"\nAll indexed files:\n")
resume_files = []

for i, metadata in enumerate(results['metadatas'], 1):
    filename = metadata.get('filename', 'Unknown')
    print(f"{i}. {filename}")
    
    # Check if it's a resume
    if 'resume' in filename.lower() or 'cv' in filename.lower():
        resume_files.append(filename)

print("\n" + "="*60)
print(f"\nFiles with 'resume' or 'cv' in name:")
if resume_files:
    for f in resume_files:
        print(f"  - {f}")
else:
    print("  None found!")