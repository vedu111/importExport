const express = require('express');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const fs = require('fs');
const path = require('path');
const bodyParser = require('body-parser');
const pdfParse = require('pdf-parse');
const dotenv = require('dotenv');
const workerpool = require('workerpool');
const multer = require('multer');
const { v4: uuidv4 } = require('uuid');

dotenv.config();

// Initialize Express app and router
const app = express();
const router = express.Router();
const port = process.env.PORT || 3000;

// File paths and API keys
const PDF_PATH = path.join(__dirname, 'merge_pdf.pdf');
const EMBEDDINGS_PATH = path.join(__dirname, 'embeddings-database.json');
const ITEM_TO_HS_PATH = path.join(__dirname, 'USA-item-to-hs-mapping.json');

// Load in-memory embeddings database if available
let embeddingsDatabase = {};
try {
  embeddingsDatabase = require('./embeddings-database.json');
} catch (err) {
  console.warn('Embeddings database not found, starting with empty database.');
}

// Set up Gemini API
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "AIzaSyAGwF77rylskhbDu4WLNf0zSWTuVlNbr5A";
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
const embeddingModel = genAI.getGenerativeModel({ model: "embedding-001" });

// Configure middleware
app.use(bodyParser.json());

// Configure multer for file uploads (allow any file type)
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/');
  },
  filename: function (req, file, cb) {
    const uniqueFileName = `${uuidv4()}-${file.originalname}`;
    cb(null, uniqueFileName);
  }
});

const upload = multer({
  storage: storage,
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

/**
 * Function to extract HS Codes from the PDF text.
 */
function extractHSCodes(pdfText) {
  const hsCodeRegex = /(\d{8})\s+(.*?)(?:\s+)(Free|Restricted|Prohibited|Not Permitted)/gi;
  const hsCodesData = {};
  const itemToHsMap = {};

  let match;
  while ((match = hsCodeRegex.exec(pdfText)) !== null) {
    const hsCode = match[1];
    const description = match[2].trim();
    const exportPolicy = match[3];

    hsCodesData[hsCode] = {
      description,
      policy: exportPolicy
    };

    const items = description.split(/[,;\/]/).map(item => item.trim().toLowerCase());
    items.forEach(item => {
      if (item.length > 3) {
        itemToHsMap[item] = hsCode;
      }
    });
    itemToHsMap[description.toLowerCase()] = hsCode;
  }
  return { hsCodesData, itemToHsMap };
}

/**
 * POST /api/upload-rules-pdf
 * Accepts any file upload. If the file is a PDF, it processes it (extracting text, HS codes, chunking,
 * and generating embeddings). For non-PDF files, it simply returns a success message.
 */
router.post('/api/upload-rules-pdf', upload.any(), async (req, res) => {
  // Check that at least one file was uploaded
  if (!req.files || req.files.length === 0) {
    return res.status(400).json({
      status: false,
      error: "No file uploaded"
    });
  }

  // Use the first file uploaded
  const file = req.files[0];

  // If the file is not a PDF, skip processing and return a message.
  if (file.mimetype !== 'application/pdf') {
    return res.json({
      status: true,
      message: "File uploaded successfully. Note: Processing is only applied to PDF files."
    });
  }

  try {
    console.log(`Processing uploaded PDF: ${file.filename}`);
    const pdfPath = file.path;

    // Read and parse the PDF file
    const dataBuffer = fs.readFileSync(pdfPath);
    const pdfData = await pdfParse(dataBuffer);
    const pdfText = pdfData.text;
    console.log('Extracted PDF Text length:', pdfText.length);

    // Extract new HS codes and item-to-HS mappings from PDF text
    const { hsCodesData: newHsCodesData, itemToHsMap: newItemToHsMap } = extractHSCodes(pdfText);

    // Load existing data if available
    let existingEmbeddingsData = {};
    let existingItemToHsMap = {};
    if (fs.existsSync(EMBEDDINGS_PATH)) {
      existingEmbeddingsData = JSON.parse(fs.readFileSync(EMBEDDINGS_PATH, 'utf8'));
    }
    if (fs.existsSync(ITEM_TO_HS_PATH)) {
      existingItemToHsMap = JSON.parse(fs.readFileSync(ITEM_TO_HS_PATH, 'utf8'));
    }

    // Merge HS code data and mappings
    const mergedHsCodesData = {
      ...(existingEmbeddingsData.hsCodesData || {}),
      ...newHsCodesData
    };
    const mergedItemToHsMap = {
      ...existingItemToHsMap,
      ...newItemToHsMap
    };

    // Process the PDF text into chunks
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

    // Create a worker pool for parallel embedding generation
    const pool = workerpool.pool(path.join(__dirname, 'worker.js'));
    console.log(`Generating embeddings for ${textChunks.length} chunks from uploaded PDF...`);

    let completed = 0;
    const total = textChunks.length;
    const embeddingPromises = textChunks.map(async (chunk, index) => {
      try {
        const embedding = await pool.exec('generateEmbedding', [chunk]);
        completed++;
        if (completed % 5 === 0) {
          console.log(`Processed ${completed} out of ${total} chunks`);
        }
        return {
          id: index + (existingEmbeddingsData.chunks?.length || 0),
          content: chunk,
          embedding
        };
      } catch (error) {
        console.error(`Error generating embedding for chunk ${index}:`, error);
        return null;
      }
    });

    const results = await Promise.all(embeddingPromises);
    const successfulChunks = results.filter(result => result !== null);

    // Merge new chunks with existing ones and update the embeddings database
    const mergedChunks = [
      ...(existingEmbeddingsData.chunks || []),
      ...successfulChunks
    ];
    const updatedEmbeddingsDatabase = {
      chunks: mergedChunks,
      hsCodesData: mergedHsCodesData
    };

    fs.writeFileSync(EMBEDDINGS_PATH, JSON.stringify(updatedEmbeddingsDatabase, null, 2));
    fs.writeFileSync(ITEM_TO_HS_PATH, JSON.stringify(mergedItemToHsMap, null, 2));

    // Update the in-memory database
    embeddingsDatabase.chunks = updatedEmbeddingsDatabase.chunks;
    embeddingsDatabase.hsCodesData = updatedEmbeddingsDatabase.hsCodesData;
    embeddingsDatabase.itemToHsMap = mergedItemToHsMap;

    // Terminate the worker pool and remove the temporary file
    await pool.terminate();
    fs.unlinkSync(pdfPath);

    return res.json({
      status: true,
      message: "PDF processed and merged successfully",
      stats: {
        newHSCodesAdded: Object.keys(newHsCodesData).length,
        newItemMappingsAdded: Object.keys(newItemToHsMap).length,
        newTextChunksProcessed: successfulChunks.length,
        totalHSCodes: Object.keys(mergedHsCodesData).length,
        totalChunks: mergedChunks.length
      }
    });
  } catch (error) {
    console.error('Error processing uploaded PDF:', error);
    if (req.files && req.files[0] && fs.existsSync(req.files[0].path)) {
      fs.unlinkSync(req.files[0].path);
    }
    return res.status(500).json({
      status: false,
      error: "An error occurred while processing the uploaded PDF",
      message: error.message
    });
  }
});

/**
 * Helper Functions
 */
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function findHSCodeByItemName(itemName, itemToHsMap) {
  const normalizedItemName = itemName.toLowerCase().trim();
  if (itemToHsMap[normalizedItemName]) {
    return itemToHsMap[normalizedItemName];
  }
  const itemKeys = Object.keys(itemToHsMap);
  const containsMatch = itemKeys.find(key => key.includes(normalizedItemName));
  if (containsMatch) {
    return itemToHsMap[containsMatch];
  }
  const isContainedMatch = itemKeys.find(key => normalizedItemName.includes(key) && key.length > 5);
  return isContainedMatch ? itemToHsMap[isContainedMatch] : null;
}

function findHSCodeByDescription(description, hsCodesData) {
  const normalizedDescription = description.toLowerCase().trim();
  const matchingHsCode = Object.keys(hsCodesData).find(hsCode =>
    hsCodesData[hsCode].description.toLowerCase() === normalizedDescription
  );
  if (matchingHsCode) return matchingHsCode;
  const partialMatchHsCode = Object.keys(hsCodesData).find(hsCode =>
    hsCodesData[hsCode].description.toLowerCase().includes(normalizedDescription) ||
    normalizedDescription.includes(hsCodesData[hsCode].description.toLowerCase())
  );
  return partialMatchHsCode || null;
}

async function checkHSCodeCompliance(hsCode, embeddingsDatabase) {
  if (embeddingsDatabase.hsCodesData && embeddingsDatabase.hsCodesData[hsCode]) {
    const hsData = embeddingsDatabase.hsCodesData[hsCode];
    return {
      exists: true,
      allowed: hsData.policy.toLowerCase() === 'free',
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
      if (policies.some(policy => policy.toLowerCase() === 'free')) {
        return {
          exists: true,
          allowed: true,
          policy: 'Free',
          description: `Falls under chapter ${chapter} which has some free categories`
        };
      } else {
        return {
          exists: true,
          allowed: false,
          policy: policies[0],
          description: `Falls under chapter ${chapter} which has no free categories`
        };
      }
    }
  }
  try {
    const prompt = `Given HS code ${hsCode} that wasn't found in our database, provide a reason why this code might not be recognized. Limit your response to one short paragraph.`;
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
      reason: `The HS Code ${hsCode} was not found in the export compliance regulations. Please verify the code and try again.`
    };
  }
}

async function findRelevantContent(query, embeddingsDatabase) {
  try {
    const queryEmbedResult = await embeddingModel.embedContent({
      content: { parts: [{ text: query }] }
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

/**
 * POST /api/check-export-compliance
 * Checks export compliance based on provided HS code, item name, or item description.
 */
router.post('/api/check-export-compliance', async (req, res) => {
  try {
    const { hsCode, itemWeight, material, itemName, itemManufacturer, itemDescription } = req.body;
    if (!hsCode && !itemName && !itemDescription) {
      return res.status(400).json({
        status: false,
        error: "Missing required fields. Please provide either hsCode, itemName, or itemDescription"
      });
    }
    let codeToCheck = hsCode;
    if (!hsCode && itemName) {
      codeToCheck = findHSCodeByItemName(itemName, embeddingsDatabase.itemToHsMap);
      if (!codeToCheck) {
        return res.json({
          status: false,
          allowed: false,
          reason: `Could not find an HS code matching item name: ${itemName}. Please provide a valid HS code.`
        });
      }
    }
    if (!hsCode && itemDescription) {
      const normalizedDescription = itemDescription.toLowerCase().trim();
      codeToCheck = findHSCodeByDescription(normalizedDescription, embeddingsDatabase.hsCodesData);
      if (!codeToCheck) {
        try {
          const prompt = `Given the description "${normalizedDescription}" not found in USA import regulations, provide a short reason why this item is restricted for export.`;
          const result = await model.generateContent({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: { maxOutputTokens: 100 }
          });
          return res.json({
            status: false,
            allowed: false,
            reason: result.response.text(),
            queriedDescription: normalizedDescription
          });
        } catch (aiError) {
          console.error('Error generating reason with AI:', aiError);
          return res.json({
            status: false,
            allowed: false,
            reason: `No matching HS code found for description "${normalizedDescription}". Unable to determine a specific reason due to an AI processing error.`,
            queriedDescription: normalizedDescription
          });
        }
      }
    }
    const hsCodeCompliance = await checkHSCodeCompliance(codeToCheck, embeddingsDatabase);
    if (!hsCodeCompliance.exists) {
      return res.json({
        status: false,
        allowed: false,
        reason: hsCodeCompliance.reason,
        queriedHsCode: codeToCheck,
        queriedItemName: itemName || null
      });
    }
    if (hsCodeCompliance.allowed) {
      return res.json({
        status: true,
        allowed: true,
        hsCode: codeToCheck,
        policy: hsCodeCompliance.policy,
        description: hsCodeCompliance.description,
        conditions: "Standard export conditions apply",
        queriedItemName: itemName || null
      });
    } else {
      try {
        const prompt = `Given the HS code ${codeToCheck} is not allowed for export, provide a short reason why this item is restricted.`;
        const result = await model.generateContent({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: { maxOutputTokens: 100 }
        });
        return res.json({
          status: false,
          allowed: false,
          hsCode: codeToCheck,
          policy: hsCodeCompliance.policy,
          description: hsCodeCompliance.description,
          reason: result.response.text(),
          queriedItemName: itemName || null
        });
      } catch (aiError) {
        console.error('Error generating reason with AI:', aiError);
        return res.json({
          status: false,
          allowed: false,
          hsCode: codeToCheck,
          policy: hsCodeCompliance.policy,
          description: hsCodeCompliance.description,
          reason: `Export not allowed for HS Code ${codeToCheck} with policy ${hsCodeCompliance.policy}. Unable to determine a reason due to an AI processing error.`,
          queriedItemName: itemName || null
        });
      }
    }
  } catch (error) {
    console.error('Error checking export compliance:', error);
    return res.status(500).json({
      status: false,
      error: "An error occurred while checking export compliance"
    });
  }
});

/**
 * POST /api/find-by-description
 * Finds an HS code based on the provided item description.
 */
router.post('/api/find-by-description', (req, res) => {
  try {
    const { description } = req.body;
    if (!description) {
      return res.status(400).json({
        status: false,
        error: "Missing required field: description"
      });
    }
    const normalizedDescription = description.toLowerCase().trim();
    const hsCodesData = embeddingsDatabase.hsCodesData || {};
    const matchingHsCode = Object.keys(hsCodesData).find(hsCode =>
      hsCodesData[hsCode].description.toLowerCase() === normalizedDescription
    );
    if (matchingHsCode) {
      return res.json({ hsCode: matchingHsCode });
    }
    const partialMatchHsCode = Object.keys(hsCodesData).find(hsCode =>
      hsCodesData[hsCode].description.toLowerCase().includes(normalizedDescription) ||
      normalizedDescription.includes(hsCodesData[hsCode].description.toLowerCase())
    );
    if (partialMatchHsCode) {
      return res.json({ status: true, hsCode: partialMatchHsCode, note: "Found via partial match" });
    }
    return res.json({ status: false, error: "No matching HS code found for this description" });
  } catch (error) {
    console.error('Error finding HS code by description:', error);
    return res.status(500).json({ status: false, error: "An error occurred while finding HS code" });
  }
});

// Mount the router and start the server
app.use(router);
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

module.exports = router;
