⚛️ PhysicsBot – AI Chatbot for Physics Q&A

PhysicsBot is an AI-powered chatbot designed to answer physics-related questions based on The Feynman Lectures on Physics (Vol. 1, 2, 3). It processes the book, stores vectorized data in a free open-source vector database, and uses a Large Language Model (LLM) to generate accurate responses.

✨ Features
     1.	Text Processing: Extracts and cleans text from The Feynman Lectures on Physics.
     2.	Vector Storage: Embeds processed text and stores it in an open-source vector database for 	efficient retrieval.
     3.	LLM-Based Responses: Uses a lightweight, open-source NLP model to generate context-aware 	answers.
     4.	Semantic Search: Retrieves the most relevant sections from the book based on user queries.
     5.	Scalable & Efficient: Optimized for quick responses while keeping resource usage minimal.

🔍 Workflow
     1.	Data Processing: Extracts text from The Feynman Lectures on Physics PDF.
     2.	Vectorization: Converts extracted text into embeddings and stores them in a vector database.
     3.	User Query Processing: Accepts natural language questions from users.
     4.	Retrieval & Response Generation: Fetches the most relevant sections and generates an AI-	powered response.

🏗️ Code Structure
     1.	Data Processing: Parses and cleans the book’s text.
     2.	Embedding & Storage: Generates vector embeddings and stores them in a vector database.
     3.	Query Handling: Processes user input and retrieves relevant sections.
     4.	Response Generation: Uses an LLM to generate physics-based answers.
     5.	UI Integration: Can be extended with a chatbot interface (e.g., Streamlit,).
