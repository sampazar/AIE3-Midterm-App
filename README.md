---
title: Midterm App
emoji: 🏠
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# Midterm App

This is the Midterm App, a project developed for the AI Engineering course. The application leverages Chainlit, LangChain, OpenAI, and Qdrant to perform retrieval-augmented generation (RAG) on a PDF document.

## Features

- **Document Loading**: Loads and splits a PDF document into manageable chunks.
- **Embeddings and Retrieval**: Uses OpenAI embeddings and Qdrant for efficient document retrieval.
- **Question Answering**: Answers questions based on the context retrieved from the document.
- **Chainlit Integration**: Provides a chat interface for users to interact with the application.

## Setup

To set up and run the application locally, follow these steps:

1. **Clone the repository**:
   git clone https://huggingface.co/spaces/sampazar/midterm-app
   cd midterm-app
2. Build and run the Docker container:
    docker build -t midterm-app .
    docker run -p 7860:7860 midterm-app

## Requirements
Make sure you have the following dependencies installed if you are not using Docker:
    Python 3.9
    pip

## Dependencies
The application depends on several Python packages, which are listed in the requirements.txt file. You can install them using:
    pip install -r requirements.txt

## Usage
Run the application with: 
    chainlit run app.py --port 7860

Once the application is running, you can access it in your browser at http://localhost:7860.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.