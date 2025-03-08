// worker.js
const { GoogleGenerativeAI } = require('@google/generative-ai');
const workerpool = require('workerpool');
const dotenv = require('dotenv');
const path = require('path');

// Load environment variables from .env file in the same directory
dotenv.config({ path: path.join(__dirname, '.env') });

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const embeddingModel = genAI.getGenerativeModel({ model: "embedding-001" });

async function generateEmbedding(chunk) {
  const embedResult = await embeddingModel.embedContent({
    content: { parts: [{ text: chunk }] },
  });
  return embedResult.embedding.values;
}

// Expose the function to the worker pool
workerpool.worker({
  generateEmbedding: generateEmbedding
});