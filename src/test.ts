// Test Suite for GodsEye Smart MCP Intent Detection
// Run this to validate the routing logic

import { analyzeIntent } from './godseye-smart-mcp.js';

interface TestCase {
  input: string;
  expectedZone: string;
  expectedQueryFilter?: string;
  description: string;
}

const testCases: TestCase[] = [
  // ===== STRATEGIST ZONE TESTS =====
  {
    input: "How is my product performing?",
    expectedZone: "strategist",
    description: "Basic performance question"
  },
  {
    input: "What's my SOV score?",
    expectedZone: "strategist",
    description: "Direct score request"
  },
  {
    input: "Give me a summary of competitor gaps",
    expectedZone: "strategist",
    description: "Summary with competitor mention"
  },
  {
    input: "Show me overall visibility metrics",
    expectedZone: "strategist",
    description: "Overview request"
  },
  {
    input: "What are the biggest threats to my market position?",
    expectedZone: "strategist",
    description: "Strategic threats question"
  },

  // ===== DETECTIVE ZONE TESTS =====
  {
    input: "Why did I lose the query for 'best CRM'?",
    expectedZone: "detective",
    expectedQueryFilter: "best CRM",
    description: "Specific loss with query"
  },
  {
    input: "Who is winning the 'pricing' queries?",
    expectedZone: "detective",
    expectedQueryFilter: "pricing",
    description: "Winner question with keyword"
  },
  {
    input: "Show me query insights for customer support",
    expectedZone: "detective",
    description: "Query insights request"
  },
  {
    input: "Debug my search visibility for 'enterprise features'",
    expectedZone: "detective",
    expectedQueryFilter: "enterprise features",
    description: "Debug request with specific query"
  },
  {
    input: "Why am I not showing up for startup queries?",
    expectedZone: "detective",
    description: "Visibility loss question"
  },

  // ===== ARCHITECT ZONE TESTS =====
  {
    input: "Get me the AEO plan",
    expectedZone: "architect",
    description: "Direct plan request"
  },
  {
    input: "I need the full DNA blueprint",
    expectedZone: "architect",
    description: "Blueprint request"
  },
  {
    input: "Generate the optimization rules",
    expectedZone: "architect",
    description: "Optimization rules request"
  },
  {
    input: "Show me the complete AEO architecture",
    expectedZone: "architect",
    description: "Architecture request"
  },

  // ===== MULTI-ZONE TESTS =====
  {
    input: "Give me a comprehensive analysis with all available data",
    expectedZone: "multi",
    description: "Explicit comprehensive request"
  },
  {
    input: "I need everything - strategic overview, query details, and the plan",
    expectedZone: "multi",
    description: "Multi-zone explicit"
  },
  {
    input: "Full audit with scores and optimization recommendations",
    expectedZone: "multi",
    description: "Audit requiring multiple zones"
  },

  // ===== AMBIGUOUS/EDGE CASES =====
  {
    input: "Help me with SEO",
    expectedZone: "strategist", // Default to strategist for ambiguous
    description: "Vague SEO help"
  },
  {
    input: "My rankings are down",
    expectedZone: "detective", // Could be detective to find why
    description: "General problem statement"
  },
  {
    input: "How do I improve my visibility for 'API integration'?",
    expectedZone: "multi", // Needs strategy + detective + plan
    expectedQueryFilter: "API integration",
    description: "Improvement question (needs multiple zones)"
  }
];

console.log("🧪 GodsEye Smart MCP - Intent Detection Test Suite\n");
console.log("=" .repeat(70));

let passed = 0;
let failed = 0;

testCases.forEach(async (test, index) => {
  console.log(`\nTest ${index + 1}: ${test.description}`);
  console.log(`Input: "${test.input}"`);
  
  const result = analyzeIntent(test.input);
  
  console.log(`Expected Zone: ${test.expectedZone.toUpperCase()}`);
  console.log(`Detected Zone: ${(await result).zones.join(', ').toUpperCase()}`);
  
  const zoneMatch = (await result).zones.includes(test.expectedZone as any);
  const queryMatch = !test.expectedQueryFilter || (await result).queryFilter === test.expectedQueryFilter;
  
  if (test.expectedQueryFilter) {
    console.log(`Expected Query Filter: "${test.expectedQueryFilter}"`);
    console.log(`Detected Query Filter: ${(await result).queryFilter || 'none'}`);
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
});

console.log(`\n${"=".repeat(70)}`);
console.log(`\n📊 Test Results: ${passed} passed, ${failed} failed out of ${testCases.length} tests`);
console.log(`Success Rate: ${((passed / testCases.length) * 100).toFixed(1)}%\n`);

if (failed === 0) {
  console.log("🎉 All tests passed! Intent detection is working correctly.\n");
} else {
  console.log("⚠️  Some tests failed. Review the intent detection patterns.\n");
}

// ===== DETAILED ANALYSIS EXAMPLES =====

console.log("\n" + "=".repeat(70));
console.log("🔍 Detailed Analysis Examples\n");

const detailedExamples = [
  "Why did I lose the query for 'best project management software for startups'?",
  "Give me a strategic overview with competitor gaps and the full optimization plan",
  "How is my product doing in the enterprise segment?"
];

detailedExamples.forEach(async (example, i) => {
  console.log(`\nExample ${i + 1}: "${example}"`);
  const analysis = await analyzeIntent(example);
  console.log(JSON.stringify(analysis, null, 2));
  console.log("-".repeat(70));
});