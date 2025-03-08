const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const indiaRoutes = require('./india');
const usaRoutes = require('./usa');

const app = express();
const PORT = 3000;

app.use(bodyParser.json());
app.use('/india', indiaRoutes);
app.use('/usa', usaRoutes);

app.get('/', (req, res) => {
  res.send('Export Compliance API is running!');
});

app.post('/api/check-shipment-compliance', async (req, res) => {
    try {
      const shipmentData = req.body;
      let report = [];
      let overallStatus = true;
  
      for (const box of shipmentData.boxes) {
        for (const item of box.items) {
          let itemReport = {
            itemName: item.itemName,
            itemManufacturer: item.itemManufacturer || "Not specified",
            material: item.material || "Not specified",
            itemWeight: item.itemWeight || "Not specified",
          };
          
          // Step 1: Get or validate HS code
          let hsCode = item.hsCode;
          
          if (!hsCode) {
            try {
              // Try to find HS code based on item name
              const hsResponse = await axios.post('http://localhost:3000/india/api/find-by-description', {
                description: item.itemName
              });
              
              if (hsResponse.data.status && hsResponse.data.hsCode) {
                hsCode = hsResponse.data.hsCode;
                itemReport.hsCode = hsCode;
                itemReport.hsCodeNote = hsResponse.data.note || "Generated from item name";
              } else {
                itemReport.status = false;
                itemReport.exportStatus = false;
                itemReport.reason = hsResponse.data.reason || 'HS Code could not be determined';
                report.push(itemReport);
                overallStatus = false;
                continue;
              }
            } catch (error) {
              console.error('Error finding HS code:', error);
              itemReport.status = false;
              itemReport.exportStatus = false;
              itemReport.reason = 'Error determining HS code';
              report.push(itemReport);
              overallStatus = false;
              continue;
            }
          } else {
            itemReport.hsCode = hsCode;
          }
  
          // Step 2: Check export compliance from India
          try {
            const exportResponse = await axios.post('http://localhost:3000/india/api/check-export-compliance', {
              hsCode: hsCode,
              itemName: item.itemName,
              itemWeight: item.itemWeight,
              material: item.material,
              itemManufacturer: item.itemManufacturer,
              itemDescription: item.itemName
            });
  
            itemReport.exportStatus = exportResponse.data.allowed;
            
            if (!exportResponse.data.allowed) {
              itemReport.status = false;
              itemReport.exportReason = exportResponse.data.reason || 'Not eligible for export from India';
              report.push(itemReport);
              overallStatus = false;
              continue;
            } else {
              itemReport.exportPolicy = exportResponse.data.policy;
              itemReport.exportDescription = exportResponse.data.description;
              itemReport.exportConditions = exportResponse.data.conditions;
            }
          } catch (error) {
            console.error('Error checking export compliance:', error);
            itemReport.status = false;
            itemReport.exportStatus = false;
            itemReport.exportReason = 'Error checking export compliance';
            report.push(itemReport);
            overallStatus = false;
            continue;
          }
  
          // Step 3: Check import compliance in USA based on HS code
          try {
            const importResponse = await axios.post('http://localhost:3000/usa/api/find-by-description', {
              description: item.itemName,
              hsCode: hsCode
            });
  
            itemReport.importStatus = importResponse.data.status;
            
            if (!importResponse.data.status) {
              itemReport.status = false;
              itemReport.importReason = importResponse.data.reason || 'Not allowed for import in USA';
              report.push(itemReport);
              overallStatus = false;
              continue;
            }
          } catch (error) {
            console.error('Error checking import compliance:', error);
            itemReport.status = false;
            itemReport.importStatus = false;
            itemReport.importReason = 'Error checking import compliance';
            report.push(itemReport);
            overallStatus = false;
            continue;
          }
  
          // If we reached here, both export and import are allowed
          itemReport.status = true;
          itemReport.message = 'Eligible for export from India and import into USA';
          report.push(itemReport);
        }
      }
  
      // Generate summary
      const summary = {
        organizationName: shipmentData.organizationName,
        sourceCountry: shipmentData.sourceAddress?.country || "Not specified",
        destinationCountry: shipmentData.destinationAddress?.country || "Not specified",
        shipmentDate: shipmentData.shipmentDate,
        totalItems: report.length,
        approvedItems: report.filter(item => item.status).length,
        rejectedItems: report.filter(item => !item.status).length
      };
  
      res.json({
        status: overallStatus,
        summary: summary,
        report: report
      });
    } catch (error) {
      console.error('Error processing shipment compliance:', error);
      res.status(500).json({
        status: false,
        error: 'An error occurred while processing compliance check',
        errorMessage: error.message
      });
    }
  });
  
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});