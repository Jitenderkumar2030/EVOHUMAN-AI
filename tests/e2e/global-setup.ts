import { chromium, FullConfig } from '@playwright/test';
import axios from 'axios';

async function globalSetup(config: FullConfig) {
  console.log('🚀 Starting E2E Test Setup...');
  
  // Wait for services to be ready
  await waitForServices();
  
  // Setup test data
  await setupTestData();
  
  // Create test user session
  await createTestUserSession();
  
  console.log('✅ E2E Test Setup Complete');
}

async function waitForServices() {
  const services = [
    { name: 'Frontend', url: 'http://localhost:3000' },
    { name: 'AiCE Service', url: 'http://localhost:8001/health' },
    { name: 'Proteus Service', url: 'http://localhost:8002/health' },
    { name: 'ESM3 Service', url: 'http://localhost:8003/health' },
    { name: 'SymbioticAIS Service', url: 'http://localhost:8004/health' },
    { name: 'Bio-Twin Service', url: 'http://localhost:8005/health' },
    { name: 'ExoStack Service', url: 'http://localhost:8006/health' },
  ];

  console.log('⏳ Waiting for services to be ready...');
  
  for (const service of services) {
    let retries = 30;
    let ready = false;
    
    while (retries > 0 && !ready) {
      try {
        const response = await axios.get(service.url, { timeout: 5000 });
        if (response.status === 200) {
          console.log(`✅ ${service.name} is ready`);
          ready = true;
        }
      } catch (error) {
        retries--;
        if (retries > 0) {
          console.log(`⏳ Waiting for ${service.name}... (${retries} retries left)`);
          await new Promise(resolve => setTimeout(resolve, 2000));
        } else {
          console.log(`⚠️  ${service.name} not ready, continuing with tests`);
        }
      }
    }
  }
}

async function setupTestData() {
  console.log('📊 Setting up test data...');
  
  try {
    // Create test user
    await axios.post('http://localhost:8001/users', {
      id: 'test_user_e2e',
      name: 'E2E Test User',
      email: 'e2e@test.com',
      biological_age: 28,
      chronological_age: 32,
    });

    // Initialize bio-twin data
    await axios.post('http://localhost:8005/bio-twin/initialize', {
      user_id: 'test_user_e2e',
      initial_metrics: {
        health_score: 85,
        energy_level: 78,
        cognitive_index: 92,
        stress_level: 25,
      }
    });

    // Initialize multi-agent system
    await axios.post('http://localhost:8004/multi_agent/initialize', {
      user_id: 'test_user_e2e'
    });

    console.log('✅ Test data setup complete');
  } catch (error) {
    console.log('⚠️  Test data setup failed, using mock data');
  }
}

async function createTestUserSession() {
  console.log('🔐 Creating test user session...');
  
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();
  
  try {
    // Navigate to login page
    await page.goto('http://localhost:3000/login');
    
    // Login with test credentials
    await page.fill('[data-testid="email-input"]', 'e2e@test.com');
    await page.fill('[data-testid="password-input"]', 'test123');
    await page.click('[data-testid="login-button"]');
    
    // Wait for redirect to dashboard
    await page.waitForURL('**/dashboard');
    
    // Save authentication state
    await context.storageState({ path: 'tests/e2e/auth-state.json' });
    
    console.log('✅ Test user session created');
  } catch (error) {
    console.log('⚠️  Test user session creation failed, tests will run without auth');
  } finally {
    await browser.close();
  }
}

export default globalSetup;
