import os
from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone
from openai import OpenAI

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index('neet-assistant')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Get ALL NTA bulletin chunks
print("--- Searching ALL NTA bulletin chunks for date keywords ---")

# Use a generic query to get all chunks
query = "NEET exam"
response = client.embeddings.create(model="text-embedding-3-small", input=query)
query_vector = response.data[0].embedding

results = index.query(
    vector=query_vector,
    top_k=200,  # Get more chunks
    include_metadata=True,
    filter={'document_type': 'nta_bulletin'}
)

print(f"Got {len(results['matches'])} chunks")

# Search through all for date keywords
found_any = False
for r in results['matches']:
    m = r['metadata']
    text = m.get('text', '')
    
    # Look for specific date patterns
    if any(kw in text for kw in ['03 May', '08 February', '08 March', 'Date of Examination', 'Online Submission']):
        found_any = True
        print(f"\n*** FOUND on page {m.get('page_label')} ***")
        print(text[:800])
        print("---")

if not found_any:
    print("\n!!! NO CHUNKS contain the dates table content !!!")
    print("This means the PDF table was not extracted properly.")
