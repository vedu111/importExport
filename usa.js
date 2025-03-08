const express = require('express');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const fs = require('fs');
const path = require('path');
const bodyParser = require('body-parser');
const pdfParse = require('pdf-parse');
const dotenv = require('dotenv');
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const os = require('os');
const router = express.Router();
const embeddingsDatabase = require('./usa-embeddings-database.json');

dotenv.config();

// Initialize Express app
const app = express();
const port = process.env.PORT || 3001;

// Configure middleware
app.use(bodyParser.json());

// Set up Gemini API
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "AIzaSyAGwF77rylskhbDu4WLNf0zSWTuVlNbr5A";
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
const embeddingModel = genAI.getGenerativeModel({ model: "embedding-001" });

// Path to the PDF and embeddings file
const PDF_PATH = path.join(__dirname, 'USA.pdf');
const EMBEDDINGS_PATH = path.join(__dirname, 'usa-embeddings-database.json');
const ITEM_TO_HS_PATH = path.join(__dirname, 'usa-item-to-hs-mapping.json');
const CHUNKS_DIR = path.join(__dirname, 'chunks');

// Maximum number of parallel workers
const MAX_WORKERS = Math.max(1, os.cpus().length - 1);

// Ensure directories exist
if (!fs.existsSync(CHUNKS_DIR)) {
  fs.mkdirSync(CHUNKS_DIR, { recursive: true });
}

// Function to extract HS Codes from US Import PDF content
function extractHSCodes(pdfText) {
  // Example regex for US import regulations (adjust based on actual format)
  const hsCodeRegex = /(\d{8,10})\s+(.*?)(?:\s+)(Allowed|Restricted|Prohibited|Special License Required)/gi;
  const hsCodesData = {};
  const itemToHsMap = {};
  
  let match;
  while ((match = hsCodeRegex.exec(pdfText)) !== null) {
    const hsCode = match[1];
    const description = match[2].trim();
    const importPolicy = match[3];
    
    hsCodesData[hsCode] = {
      description,
      policy: importPolicy
    };
    
    // Create a mapping from items in description to HS code
    const items = description.split(/[,;\/]/).map(item => item.trim().toLowerCase());
    items.forEach(item => {
      if (item.length > 3) { // Ignore very short terms
        itemToHsMap[item] = hsCode;
      }
    });
    
    // Also add the full description as a searchable item
    itemToHsMap[description.toLowerCase()] = hsCode;
  }
  
  // If the regex didn't match anything, try a more general approach
  if (Object.keys(hsCodesData).length === 0) {
    console.warn("The regex pattern didn't match any HS codes. Using AI to extract information...");
    return null; // We'll handle this with AI in a worker
  }
  
  return { hsCodesData, itemToHsMap };
}

// Worker for extracting HS codes with AI
function createHSCodeExtractionWorker(chunk, index, totalChunks) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(`
      const { parentPort, workerData } = require('worker_threads');
      const { GoogleGenerativeAI } = require('@google/generative-ai');
      
      async function extractHSCodesWithAI() {
        try {
          const genAI = new GoogleGenerativeAI(workerData.apiKey);
          const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
          
          const prompt = \`
            Extract all HTS/HS codes with their descriptions and import policies from the following text.
            Format the output as a JSON array of objects with fields: 
            "hsCode", "description", and "policy".
            
            Text:
            \${workerData.chunk}
          \`;
          
          const result = await model.generateContent({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: { maxOutputTokens: 4096 }
          });
          
          const responseText = result.response.text();
          
          // Try to extract JSON
          try {
            const startIdx = responseText.indexOf('[');
            const endIdx = responseText.lastIndexOf(']') + 1;
            
            if (startIdx !== -1 && endIdx !== -1) {
              const jsonStr = responseText.substring(startIdx, endIdx);
              const extractedCodes = JSON.parse(jsonStr);
              
              const hsCodesData = {};
              const itemToHsMap = {};
              
              extractedCodes.forEach(item => {
                if (item.hsCode && item.description) {
                  hsCodesData[item.hsCode] = {
                    description: item.description,
                    policy: item.policy || "Unknown"
                  };
                  
                  // Create mapping
                  const items = item.description.split(/[,;\/]/).map(term => term.trim().toLowerCase());
                  items.forEach(term => {
                    if (term.length > 3) {
                      itemToHsMap[term] = item.hsCode;
                    }
                  });
                  
                  itemToHsMap[item.description.toLowerCase()] = item.hsCode;
                }
              });
              
              parentPort.postMessage({ hsCodesData, itemToHsMap });
            } else {
              parentPort.postMessage({ hsCodesData: {}, itemToHsMap: {} });
            }
          } catch (error) {
            console.error("Error parsing AI-generated JSON:", error);
            parentPort.postMessage({ hsCodesData: {}, itemToHsMap: {} });
          }
        } catch (error) {
          console.error("Worker error:", error);
          parentPort.postMessage({ hsCodesData: {}, itemToHsMap: {} });
        }
      }
      
      extractHSCodesWithAI();
    `, { eval: true, workerData: { chunk, apiKey: GEMINI_API_KEY } });

    console.log(`Started HS code extraction worker ${index + 1}/${totalChunks}`);
    
    worker.on('message', result => {
      resolve(result);
      worker.terminate();
    });
    
    worker.on('error', err => {
      console.error(`Worker ${index} error:`, err);
      reject(err);
      worker.terminate();
    });
    
    worker.on('exit', code => {
      if (code !== 0) {
        reject(new Error(`Worker ${index} stopped with exit code ${code}`));
      }
    });
  });
}

// Worker for generating embeddings
function createEmbeddingWorker(chunk, chunkId) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(`
      const { parentPort, workerData } = require('worker_threads');
      const { GoogleGenerativeAI } = require('@google/generative-ai');
      const fs = require('fs');
      const path = require('path');
      
      async function generateEmbedding() {
        try {
          const genAI = new GoogleGenerativeAI(workerData.apiKey);
          const embeddingModel = genAI.getGenerativeModel({ model: "embedding-001" });
          
          const embedResult = await embeddingModel.embedContent({
            content: { parts: [{ text: workerData.chunk }] },
          });
          
          const embedding = embedResult.embedding.values;
          
          const result = {
            id: workerData.chunkId,
            content: workerData.chunk,
            embedding: embedding
          };
          
          // Save to disk to avoid memory issues with large embeddings
          fs.writeFileSync(
            path.join(workerData.chunksDir, \`chunk_\${workerData.chunkId}.json\`),
            JSON.stringify(result)
          );
          
          parentPort.postMessage({ success: true, chunkId: workerData.chunkId });
        } catch (error) {
          console.error("Worker error:", error);
          parentPort.postMessage({ success: false, error: error.message });
        }
      }
      
      generateEmbedding();
    `, { eval: true, workerData: { 
      chunk, 
      chunkId, 
      apiKey: GEMINI_API_KEY,
      chunksDir: CHUNKS_DIR
    }});
    
    worker.on('message', result => {
      resolve(result);
      worker.terminate();
    });
    
    worker.on('error', err => {
      console.error(`Embedding worker ${chunkId} error:`, err);
      reject(err);
      worker.terminate();
    });
    
    worker.on('exit', code => {
      if (code !== 0) {
        reject(new Error(`Embedding worker ${chunkId} stopped with exit code ${code}`));
      }
    });
  });
}

// Fallback function to extract HS codes using AI when regex fails
async function extractHSCodesWithAI(pdfText) {
  try {
    const hsCodesData = {};
    const itemToHsMap = {};
    
    // Break the text into manageable chunks
    const chunks = [];
    const chunkSize = 10000;
    
    for (let i = 0; i < pdfText.length; i += chunkSize) {
      chunks.push(pdfText.substring(i, i + chunkSize));
    }
    
    // Process chunks in parallel with workers
    console.log(`Processing ${chunks.length} chunks using ${Math.min(MAX_WORKERS, chunks.length)} workers...`);
    
    const results = await processInBatches(chunks, createHSCodeExtractionWorker, MAX_WORKERS);
    
    // Merge results
    results.forEach(result => {
      Object.assign(hsCodesData, result.hsCodesData);
      Object.assign(itemToHsMap, result.itemToHsMap);
    });
    
    return { hsCodesData, itemToHsMap };
  } catch (error) {
    console.error("Error using AI to extract HS codes:", error);
    return { hsCodesData: {}, itemToHsMap: {} };
  }
}

// Function to process tasks in batches to control concurrency
async function processInBatches(items, workerFn, batchSize) {
  const results = [];
  
  for (let i = 0; i < items.length; i += batchSize) {
    const batch = items.slice(i, i + batchSize);
    const batchPromises = batch.map((item, index) => 
      workerFn(item, i + index, items.length)
    );
    
    console.log(`Processing batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(items.length/batchSize)}`);
    const batchResults = await Promise.all(batchPromises);
    results.push(...batchResults);
    
    // Small delay to avoid API rate limits
    if (i + batchSize < items.length) {
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  }
  
  return results;
}

// Function to generate embeddings and extract HS codes from PDF
async function generatePdfEmbeddings() {
  try {
    if (fs.existsSync(EMBEDDINGS_PATH) && fs.existsSync(ITEM_TO_HS_PATH)) {
      console.log('Embeddings files already exist. Using existing data.');
      const embeddingsData = JSON.parse(fs.readFileSync(EMBEDDINGS_PATH, 'utf8'));
      const itemToHsMap = JSON.parse(fs.readFileSync(ITEM_TO_HS_PATH, 'utf8'));
      return { ...embeddingsData, itemToHsMap };
    }

    console.log('Generating embeddings from PDF...');
    
    // Use a streaming approach for PDF parsing to handle large files
    const dataBuffer = fs.readFileSync(PDF_PATH);
    const pdfData = await pdfParse(dataBuffer);
    const pdfText = pdfData.text;
    
    console.log('Extracted PDF Text length:', pdfText.length);
    
    // Extract HS codes first
    console.log('Extracting HS codes from PDF...');
    let hsCodesData = {};
    let itemToHsMap = {};
    
    // Try regex first, fall back to AI if needed
    const extractionResult = extractHSCodes(pdfText);
    if (extractionResult) {
      hsCodesData = extractionResult.hsCodesData;
      itemToHsMap = extractionResult.itemToHsMap;
    } else {
      console.log('Regex extraction failed, using AI to extract HS codes...');
      const aiExtraction = await extractHSCodesWithAI(pdfText);
      hsCodesData = aiExtraction.hsCodesData;
      itemToHsMap = aiExtraction.itemToHsMap;
    }
    
    // Save item to HS code mapping
    fs.writeFileSync(ITEM_TO_HS_PATH, JSON.stringify(itemToHsMap, null, 2));
    console.log('Item to HS code mapping saved to file.');
    
    // Create text chunks for embeddings
    const chunkSize = 1000;
    const textChunks = [];
    
    const paragraphs = pdfText.split('\n\n');
    let currentChunk = '';
    
    for (const paragraph of paragraphs) {
      if ((currentChunk + paragraph).length > chunkSize) {
        if (currentChunk.length > 0) {
          textChunks.push(currentChunk.trim());
          currentChunk = '';
        }
        
        if (paragraph.length > chunkSize) {
          const words = paragraph.split(' ');
          let subChunk = '';
          
          for (const word of words) {
            if ((subChunk + ' ' + word).length > chunkSize) {
              textChunks.push(subChunk.trim());
              subChunk = word;
            } else {
              subChunk += ' ' + word;
            }
          }
          
          if (subChunk.length > 0) {
            currentChunk = subChunk.trim();
          }
        } else {
          currentChunk = paragraph;
        }
      } else {
        currentChunk += '\n\n' + paragraph;
      }
    }
    
    if (currentChunk.length > 0) {
      textChunks.push(currentChunk.trim());
    }
    
    // Clear previous chunk files
    const chunkFiles = fs.readdirSync(CHUNKS_DIR).filter(file => file.startsWith('chunk_'));
    for (const file of chunkFiles) {
      fs.unlinkSync(path.join(CHUNKS_DIR, file));
    }
    
    // Generate embeddings in parallel using workers
    console.log(`Generating embeddings for ${textChunks.length} chunks using ${Math.min(MAX_WORKERS, textChunks.length)} workers...`);
    
    const embedResults = await processInBatches(
      textChunks, 
      createEmbeddingWorker,
      MAX_WORKERS
    );
    
    // Check for any errors
    const failedChunks = embedResults.filter(result => !result.success);
    if (failedChunks.length > 0) {
      console.error(`Failed to generate embeddings for ${failedChunks.length} chunks`);
    }
    
    // Load and combine all chunk files
    const embeddingsDatabase = {
      chunks: [],
      hsCodesData: hsCodesData
    };
    
    const generatedChunkFiles = fs.readdirSync(CHUNKS_DIR)
      .filter(file => file.startsWith('chunk_'))
      .sort((a, b) => {
        const numA = parseInt(a.replace('chunk_', '').replace('.json', ''));
        const numB = parseInt(b.replace('chunk_', '').replace('.json', ''));
        return numA - numB;
      });
    
    for (const file of generatedChunkFiles) {
      const chunkData = JSON.parse(fs.readFileSync(path.join(CHUNKS_DIR, file), 'utf8'));
      embeddingsDatabase.chunks.push(chunkData);
    }
    
    fs.writeFileSync(EMBEDDINGS_PATH, JSON.stringify(embeddingsDatabase, null, 2));
    console.log('Embeddings saved to file.');
    
    return { ...embeddingsDatabase, itemToHsMap };
  } catch (error) {
    console.error('Error generating embeddings:', error);
    throw error;
  }
}

// Function to find relevant content using embeddings
async function findRelevantContent(query, embeddingsDatabase) {
  try {
    const queryEmbedResult = await embeddingModel.embedContent({
      content: { parts: [{ text: query }] },
    });
    
    const queryEmbedding = queryEmbedResult.embedding.values;
    
    const similarityScores = embeddingsDatabase.chunks.map(item => {
      const similarity = cosineSimilarity(queryEmbedding, item.embedding);
      return { ...item, similarity };
    });
    
    const topResults = similarityScores
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 5);
    
    return topResults.map(item => item.content).join('\n\n');
  } catch (error) {
    console.error('Error finding relevant content:', error);
    throw error;
  }
}

// Function to find HS code by item name
function findHSCodeByItemName(itemName, itemToHsMap) {
  const normalizedItemName = itemName.toLowerCase().trim();
  
  // Direct match
  if (itemToHsMap[normalizedItemName]) {
    return itemToHsMap[normalizedItemName];
  }
  
  // Partial match
  const itemKeys = Object.keys(itemToHsMap);
  
  // Check if item name is contained in any key
  const containsMatch = itemKeys.find(key => key.includes(normalizedItemName));
  if (containsMatch) {
    return itemToHsMap[containsMatch];
  }
  
  // Check if any key is contained in item name
  const isContainedMatch = itemKeys.find(key => normalizedItemName.includes(key) && key.length > 5);
  if (isContainedMatch) {
    return itemToHsMap[isContainedMatch];
  }
  
  return null;
}

async function checkHSCodeCompliance(hsCode, embeddingsDatabase) {
  if (embeddingsDatabase.hsCodesData && embeddingsDatabase.hsCodesData[hsCode]) {
    const hsData = embeddingsDatabase.hsCodesData[hsCode];
    return {
      exists: true,
      allowed: hsData.policy.toLowerCase() === 'allowed',
      policy: hsData.policy,
      description: hsData.description
    };
  }
  
  if (hsCode.length >= 4) {
    const chapter = hsCode.substring(0, 4);
    const twoDigitChapter = hsCode.substring(0, 2);
    
    const matchingCodes = Object.keys(embeddingsDatabase.hsCodesData || {})
      .filter(code => code.startsWith(chapter) || code.startsWith(twoDigitChapter));
    
    if (matchingCodes.length > 0) {
      const policies = matchingCodes.map(code => embeddingsDatabase.hsCodesData[code].policy);
      
      if (policies.some(policy => policy.toLowerCase() === 'allowed')) {
        return {
          exists: true,
          allowed: true,
          policy: 'Allowed',
          description: `Falls under chapter ${chapter} which has some allowed categories`
        };
      } else {
        return {
          exists: true,
          allowed: false,
          policy: policies[0],
          description: `Falls under chapter ${chapter} which has no allowed categories`
        };
      }
    }
  }
  
  try {
    // Use Gemini to generate a reason
    const prompt = `Given HTS code ${hsCode} that wasn't found in our USA import regulations database, provide a reason why this code might not be recognized or why the item might have import restrictions. Limit your response to one short paragraph.`;
    
    const result = await model.generateContent({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { maxOutputTokens: 100 }
    });
    
    const dynamicReason = result.response.text();
    
    return {
      exists: false,
      allowed: false,
      reason: dynamicReason
    };
  } catch (error) {
    console.error('Error generating dynamic reason:', error);
    return {
      exists: false,
      allowed: false,
      reason: `The HTS Code ${hsCode} was not found in the USA import regulations. Please verify the code and try again.`
    };
  }
}

// Helper function to calculate cosine similarity
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  
  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);
  
  return dotProduct / (normA * normB);
}

// Initialize embeddings database
// let embeddingsDatabase = {
//   chunks: [],
//   hsCodesData: {},
//   itemToHsMap: {}
// };

// API endpoint to check import compliance (main API)

router.post('/api/find-by-description', async (req, res) => {
  try {
    const { description } = req.body;
    if (!description) {
      return res.status(400).json({ status: false, error: "Missing required field: description" });
    }
    
    const normalizedDescription = description.toLowerCase().trim();
    const hsCodesData = embeddingsDatabase.hsCodesData || {};

    // Ensure hsCodesData has keys to prevent unnecessary lookups
    if (Object.keys(hsCodesData).length === 0) {
      return res.json({ status: false, error: "No HS codes available in the database" });
    }

    // Try exact match first
    let matchingHsCode = Object.keys(hsCodesData).find(hsCode => 
      hsCodesData[hsCode].description.toLowerCase() === normalizedDescription
    );

    if (matchingHsCode) {
      return res.json({ status: true, hsCode: matchingHsCode });
    }

    // If no exact match, try partial match
    let partialMatchHsCode = Object.keys(hsCodesData).find(hsCode => 
      hsCodesData[hsCode].description.toLowerCase().includes(normalizedDescription) ||
      normalizedDescription.includes(hsCodesData[hsCode].description.toLowerCase())
    );

    if (partialMatchHsCode) {
      return res.json({ 
        status: true, 
        hsCode: partialMatchHsCode, 
        note: "Found via partial match" 
      });
    }

    // If no match found, generate AI reason
    try {
      const prompt = `Given the description "${normalizedDescription}" not found in USA import regulations, provide a short reason why USA might ban this item for export.`;
      const result = await model.generateContent({
        contents: [{ parts: [{ text: prompt }] }],
        generationConfig: { maxOutputTokens: 100 }
      });

      return res.json({
        status: false,
        reason: result.response.text(),
        queriedDescription: normalizedDescription
      });
    } catch (aiError) {
      console.error('Error generating reason with AI:', aiError);
      return res.json({
        status: false,
        reason: `No matching HS code found for description "${normalizedDescription}". Unable to determine a specific reason due to an AI processing error.`,
        queriedDescription: normalizedDescription
      });
    }

  } catch (error) {
    console.error('Error finding HS code by description:', error);
    return res.status(500).json({ status: false, error: "An error occurred while finding HS code" });
  }
});

module.exports = router;


// // Initialize server
// async function initServer() {
//   try {
//     embeddingsDatabase = await generatePdfEmbeddings();
    
//     app.listen(port, () => {
//       console.log(`USA Import compliance API server running on port ${port}`);
//       console.log(`Loaded ${embeddingsDatabase.chunks?.length || 0} embedded chunks from PDF`);
//       console.log(`Extracted ${Object.keys(embeddingsDatabase.hsCodesData || {}).length} HTS codes from PDF`);
//       console.log(`Created ${Object.keys(embeddingsDatabase.itemToHsMap || {}).length} item to HTS code mappings`);
//     });
//   } catch (error) {
//     console.error('Failed to initialize server:', error);
//     process.exit(1);
//   }
// }

// Start the server
// initServer();





// app.post('/api/check-import-compliance', async (req, res) => {
//   try {
//     const { hsCode, itemWeight, material, itemName, itemManufacturer, countryOfOrigin } = req.body;
    
//     if (!hsCode && !itemName) {
//       return res.status(400).json({
//         status: false,
//         error: "Missing required fields. Please provide either hsCode or itemName"
//       });
//     }
    
//     let codeToCheck = hsCode;
    
//     // If hsCode is not provided but itemName is, try to find the HS code
//     if (!hsCode && itemName) {
//       codeToCheck = findHSCodeByItemName(itemName, embeddingsDatabase.itemToHsMap);
      
//       if (!codeToCheck) {
//         return res.json({
//           status: false,
//           allowed: false,
//           reason: `Could not find an HTS code matching item name: ${itemName}. Please provide a valid HTS code.`
//         });
//       }
//     }
    
//     const hsCodeCompliance = await checkHSCodeCompliance(codeToCheck, embeddingsDatabase);
    
//     // Check for additional restrictions based on country of origin
//     let countryRestriction = null;
//     if (countryOfOrigin && hsCodeCompliance.exists && hsCodeCompliance.allowed) {
//       try {
//         const prompt = `For HTS code ${codeToCheck} (${hsCodeCompliance.description}), are there any specific import restrictions or tariffs when importing from ${countryOfOrigin} to the United States? Respond with a brief explanation.`;
        
//         const result = await model.generateContent({
//           contents: [{ parts: [{ text: prompt }] }],
//           generationConfig: { maxOutputTokens: 150 }
//         });
        
//         countryRestriction = result.response.text();
//       } catch (error) {
//         console.error('Error checking country restrictions:', error);
//       }
//     }
    
//     if (!hsCodeCompliance.exists) {
//       return res.json({
//         status: false,
//         allowed: false,
//         reason: hsCodeCompliance.reason,
//         queriedHsCode: codeToCheck,
//         queriedItemName: itemName || null
//       });
//     }
    
//     if (hsCodeCompliance.allowed) {
//       return res.json({
//         status: true,
//         allowed: true,
//         hsCode: codeToCheck,
//         policy: hsCodeCompliance.policy,
//         description: hsCodeCompliance.description,
//         countryRestriction: countryRestriction,
//         conditions: "Standard import conditions apply",
//         queriedItemName: itemName || null
//       });
//     } else {
//       return res.json({
//         status: false,
//         allowed: false,
//         hsCode: codeToCheck,
//         policy: hsCodeCompliance.policy,
//         description: hsCodeCompliance.description,
//         reason: `Import not allowed for HTS Code ${codeToCheck} with policy ${hsCodeCompliance.policy}`,
//         queriedItemName: itemName || null
//       });
//     }
    
//   } catch (error) {
//     console.error('Error checking import compliance:', error);
//     return res.status(500).json({
//       status: false,
//       error: "An error occurred while checking import compliance"
//     });
//   }
// });

// // API endpoint to find HS code by item name
// app.post('/api/find-hs-code', (req, res) => {
//   try {
//     const { itemName } = req.body;
    
//     if (!itemName) {
//       return res.status(400).json({
//         status: false,
//         error: "Missing required field: itemName"
//       });
//     }
    
//     const hsCode = findHSCodeByItemName(itemName, embeddingsDatabase.itemToHsMap);
    
//     if (hsCode) {
//       const hsCodeInfo = embeddingsDatabase.hsCodesData[hsCode];
      
//       return res.json({
//         status: true,
//         itemName,
//         hsCode,
//         description: hsCodeInfo.description,
//         policy: hsCodeInfo.policy
//       });
//     } else {
//       return res.json({
//         status: false,
//         itemName,
//         error: "No matching HTS code found for this item name"
//       });
//     }
    
//   } catch (error) {
//     console.error('Error finding HS code by item name:', error);
//     return res.status(500).json({
//       status: false,
//       error: "An error occurred while finding HTS code"
//     });
//   }
// });

// // API endpoint to get all HS codes in the database
// app.get('/api/hs-codes', (req, res) => {
//   try {
//     if (!embeddingsDatabase.hsCodesData) {
//       return res.status(404).json({
//         status: false,
//         error: "HTS Codes data not found. Please regenerate embeddings."
//       });
//     }
    
//     return res.json({
//       status: true,
//       count: Object.keys(embeddingsDatabase.hsCodesData).length,
//       hsCodes: embeddingsDatabase.hsCodesData
//     });
//   } catch (error) {
//     console.error('Error retrieving HS codes:', error);
//     return res.status(500).json({
//       status: false,
//       error: "An error occurred while retrieving HTS codes"
//     });
//   }
// });

// // Endpoint to force regeneration of embeddings
// app.post('/api/regenerate-embeddings', async (req, res) => {
//   try {
//     if (fs.existsSync(EMBEDDINGS_PATH)) {
//       fs.unlinkSync(EMBEDDINGS_PATH);
//     }
    
//     if (fs.existsSync(ITEM_TO_HS_PATH)) {
//       fs.unlinkSync(ITEM_TO_HS_PATH);
//     }
    
//     // Clear chunk files
//     const chunkFiles = fs.readdirSync(CHUNKS_DIR).filter(file => file.startsWith('chunk_'));
//     for (const file of chunkFiles) {
//       fs.unlinkSync(path.join(CHUNKS_DIR, file));
//     }
    
//     embeddingsDatabase = await generatePdfEmbeddings();
    
//     res.json({
//       success: true,
//       message: "Embeddings and item mapping regenerated successfully",
//       chunksCount: embeddingsDatabase.chunks.length,
//       hsCodesCount: Object.keys(embeddingsDatabase.hsCodesData || {}).length,
//       itemMappingsCount: Object.keys(embeddingsDatabase.itemToHsMap || {}).length
//     });
//   } catch (error) {
//     console.error('Error regenerating embeddings:', error);
//     res.status(500).json({
//       success: false,
//       error: "Failed to regenerate embeddings"
//     });
//   }
// });