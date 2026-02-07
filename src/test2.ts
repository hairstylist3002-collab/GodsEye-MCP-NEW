// corrected-test.ts
import { analyzeIntent } from './godseye-smart-mcp.js';

interface TestCase {
  input: string;
  // We now expect an array of zones, or "ANY_MULTI" to accept 2+ zones
  expectedZones: string[] | "ANY_MULTI"; 
  expectedQueryFilter?: string;
  description: string;
}

const testCases: TestCase[] = [
  // ... [Keep your Strategist/Detective/Architect tests the same] ...
  {
    input: "How is my product performing?",
    expectedZones: ["strategist"],
    description: "Basic performance question"
  },
  {
    input: "Why did I lose the query for 'best CRM'?",
    expectedZones: ["detective"],
    expectedQueryFilter: "best CRM",
    description: "Specific loss with query"
  },
  {
    input: "Get me the AEO plan",
    expectedZones: ["architect"],
    description: "Direct plan request"
  },

  // ===== UPDATED MULTI-ZONE TESTS =====
  {
    input: "Give me a comprehensive analysis with all available data",
    expectedZones: "ANY_MULTI", // Expects more than 1 zone
    description: "Explicit comprehensive request"
  },
  {
    input: "Full audit with scores and optimization recommendations",
    expectedZones: "ANY_MULTI",
    description: "Audit requiring multiple zones"
  },
  {
    input: "How do I improve my visibility for 'API integration'?",
    expectedZones: "ANY_MULTI", // Detective + Architect is valid
    expectedQueryFilter: "API integration",
    description: "Improvement question"
  }
];

// WRAPPER FUNCTION TO HANDLE ASYNC CORRECTLY
async function runTests() {
  console.log("🧪 GodsEye Smart MCP - Intent Detection Test Suite\n");
  console.log("=".repeat(70));

  let passed = 0;
  let failed = 0;

  // Use for...of loop to wait for each test
  for (const [index, test] of testCases.entries()) {
    console.log(`\nTest ${index + 1}: ${test.description}`);
    console.log(`Input: "${test.input}"`);

    const result = await analyzeIntent(test.input);

    console.log(`Detected Zones: ${result.zones.join(', ').toUpperCase()}`);
    
    // LOGIC CHECK
    let zoneMatch = false;
    
    if (test.expectedZones === "ANY_MULTI") {
      // Pass if 2 or more zones are returned
      zoneMatch = result.zones.length >= 2;
      if (!zoneMatch) console.log(`   Expected Multiple Zones, got: ${result.zones}`);
    } else {
      // Check if the expected zones are present
      zoneMatch = test.expectedZones.every(z => result.zones.includes(z as any));
    }

    const queryMatch = !test.expectedQueryFilter || 
      result.queryFilter?.toLowerCase().includes(test.expectedQueryFilter.toLowerCase());

    if (test.expectedQueryFilter) {
      console.log(`Expected Filter: "${test.expectedQueryFilter}"`);
      console.log(`Detected Filter: ${result.queryFilter || 'none'}`);
    }

    if (zoneMatch && queryMatch) {
      console.log("✅ PASS");
      passed++;
    } else {
      console.log("❌ FAIL");
      if (!zoneMatch) console.log(`   Zone mismatch!`);
      if (!queryMatch) console.log(`   Query filter mismatch!`);
      failed++;
    }
    console.log("-".repeat(70));
  }

  console.log(`\n${"=".repeat(70)}`);
  console.log(`\n📊 Test Results: ${passed} passed, ${failed} failed`);
  console.log(`Success Rate: ${((passed / testCases.length) * 100).toFixed(1)}%\n`);
}

// Run the suite
runTests();