import { test, expect } from '@playwright/test';

test.describe('Protein Analysis Interface', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/protein-analysis');
    await page.waitForLoadState('networkidle');
  });

  test('should display protein analysis interface', async ({ page }) => {
    // Check page title and header
    await expect(page).toHaveTitle(/Protein Analysis/);
    await expect(page.locator('h1')).toContainText('Protein Analysis');
    
    // Check main interface elements
    await expect(page.locator('[data-testid="protein-analysis-interface"]')).toBeVisible();
    await expect(page.locator('[data-testid="sequence-input-panel"]')).toBeVisible();
    await expect(page.locator('[data-testid="analysis-panel"]')).toBeVisible();
  });

  test('should navigate between analysis tabs', async ({ page }) => {
    const tabs = ['sequence', 'structure', 'mutations', 'evolution', 'batch'];
    
    for (const tab of tabs) {
      await page.click(`[data-testid="tab-${tab}"]`);
      await expect(page.locator(`[data-testid="tab-${tab}"]`)).toHaveClass(/border-blue-500/);
      await expect(page.locator(`[data-testid="content-${tab}"]`)).toBeVisible();
    }
  });

  test('should load example protein sequences', async ({ page }) => {
    // Check example sequences dropdown
    await expect(page.locator('[data-testid="example-sequences"]')).toBeVisible();
    
    // Select Human Insulin example
    await page.selectOption('[data-testid="example-sequences"]', '0');
    
    // Check that sequence is loaded
    const sequenceInput = page.locator('[data-testid="sequence-input"]');
    await expect(sequenceInput).not.toBeEmpty();
    
    // Check sequence length indicator
    const lengthIndicator = page.locator('[data-testid="sequence-length"]');
    await expect(lengthIndicator).toContainText(/Length: \d+ amino acids/);
  });

  test('should analyze protein sequence', async ({ page }) => {
    // Load example sequence
    await page.selectOption('[data-testid="example-sequences"]', '0');
    
    // Start analysis
    await page.click('[data-testid="analyze-button"]');
    
    // Check loading state
    await expect(page.locator('[data-testid="analyze-button"]')).toContainText('Analyzing...');
    await expect(page.locator('[data-testid="analyze-button"]')).toBeDisabled();
    
    // Wait for analysis to complete (with timeout)
    await page.waitForSelector('[data-testid="analysis-complete"]', { timeout: 30000 });
    
    // Check analysis results
    await expect(page.locator('[data-testid="analysis-summary"]')).toBeVisible();
    await expect(page.locator('[data-testid="confidence-score"]')).toBeVisible();
    await expect(page.locator('[data-testid="processing-time"]')).toBeVisible();
  });

  test('should display sequence visualization', async ({ page }) => {
    // Load example sequence and analyze
    await page.selectOption('[data-testid="example-sequences"]', '0');
    await page.click('[data-testid="analyze-button"]');
    await page.waitForSelector('[data-testid="analysis-complete"]', { timeout: 30000 });
    
    // Check sequence visualization
    await expect(page.locator('[data-testid="sequence-visualization"]')).toBeVisible();
    
    // Check amino acid coloring
    const aminoAcids = page.locator('[data-testid="amino-acid"]');
    const aaCount = await aminoAcids.count();
    expect(aaCount).toBeGreaterThan(0);
    
    // Check functional domains if present
    const domains = page.locator('[data-testid="functional-domain"]');
    const domainCount = await domains.count();
    if (domainCount > 0) {
      await expect(domains.first()).toBeVisible();
    }
  });

  test('should display 3D structure viewer', async ({ page }) => {
    // Load example sequence and analyze
    await page.selectOption('[data-testid="example-sequences"]', '0');
    await page.click('[data-testid="analyze-button"]');
    await page.waitForSelector('[data-testid="analysis-complete"]', { timeout: 30000 });
    
    // Navigate to structure tab
    await page.click('[data-testid="tab-structure"]');
    
    // Check 3D structure viewer
    await expect(page.locator('[data-testid="structure-viewer"]')).toBeVisible();
    await expect(page.locator('[data-testid="structure-canvas"]')).toBeVisible();
    
    // Check secondary structure analysis
    await expect(page.locator('[data-testid="secondary-structure"]')).toBeVisible();
    await expect(page.locator('[data-testid="alpha-helix-percentage"]')).toBeVisible();
    await expect(page.locator('[data-testid="beta-sheet-percentage"]')).toBeVisible();
    await expect(page.locator('[data-testid="random-coil-percentage"]')).toBeVisible();
    
    // Check structure quality metrics
    await expect(page.locator('[data-testid="structure-quality"]')).toBeVisible();
  });

  test('should handle custom sequence input', async ({ page }) => {
    // Enter custom sequence
    const customSequence = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK';
    
    await page.fill('[data-testid="sequence-input"]', customSequence);
    
    // Check sequence length updates
    await expect(page.locator('[data-testid="sequence-length"]')).toContainText(`Length: ${customSequence.length} amino acids`);
    
    // Analyze custom sequence
    await page.click('[data-testid="analyze-button"]');
    await page.waitForSelector('[data-testid="analysis-complete"]', { timeout: 30000 });
    
    // Check that analysis completed
    await expect(page.locator('[data-testid="analysis-summary"]')).toBeVisible();
  });

  test('should validate sequence input', async ({ page }) => {
    // Test invalid characters
    await page.fill('[data-testid="sequence-input"]', 'INVALID123SEQUENCE');
    await page.click('[data-testid="analyze-button"]');
    
    // Should show validation error
    await expect(page.locator('[data-testid="validation-error"]')).toBeVisible();
    
    // Test empty sequence
    await page.fill('[data-testid="sequence-input"]', '');
    await page.click('[data-testid="analyze-button"]');
    
    // Button should be disabled for empty sequence
    await expect(page.locator('[data-testid="analyze-button"]')).toBeDisabled();
  });

  test('should display analysis history', async ({ page }) => {
    // Perform multiple analyses
    const sequences = ['0', '1', '2']; // Example sequence indices
    
    for (const seqIndex of sequences) {
      await page.selectOption('[data-testid="example-sequences"]', seqIndex);
      await page.click('[data-testid="analyze-button"]');
      await page.waitForSelector('[data-testid="analysis-complete"]', { timeout: 30000 });
      await page.waitForTimeout(1000); // Brief pause between analyses
    }
    
    // Check analysis history
    const historyItems = page.locator('[data-testid="history-item"]');
    const historyCount = await historyItems.count();
    expect(historyCount).toBeGreaterThan(0);
    
    // Click on history item to load previous analysis
    if (historyCount > 0) {
      await historyItems.first().click();
      await expect(page.locator('[data-testid="analysis-summary"]')).toBeVisible();
    }
  });

  test('should handle mutation analysis', async ({ page }) => {
    // Load example sequence and analyze
    await page.selectOption('[data-testid="example-sequences"]', '0');
    await page.click('[data-testid="analyze-button"]');
    await page.waitForSelector('[data-testid="analysis-complete"]', { timeout: 30000 });
    
    // Navigate to mutations tab
    await page.click('[data-testid="tab-mutations"]');
    
    // Check mutation analysis interface
    await expect(page.locator('[data-testid="mutation-analysis"]')).toBeVisible();
    
    // Check for predicted mutations if available
    const mutations = page.locator('[data-testid="mutation-item"]');
    const mutationCount = await mutations.count();
    
    if (mutationCount > 0) {
      await expect(mutations.first()).toBeVisible();
      await expect(page.locator('[data-testid="mutation-effect"]').first()).toBeVisible();
    } else {
      // Check empty state
      await expect(page.locator('[data-testid="mutations-empty-state"]')).toBeVisible();
    }
  });

  test('should handle evolution analysis', async ({ page }) => {
    // Load example sequence and analyze
    await page.selectOption('[data-testid="example-sequences"]', '0');
    await page.click('[data-testid="analyze-button"]');
    await page.waitForSelector('[data-testid="analysis-complete"]', { timeout: 30000 });
    
    // Navigate to evolution tab
    await page.click('[data-testid="tab-evolution"]');
    
    // Check evolution analysis interface
    await expect(page.locator('[data-testid="evolution-analysis"]')).toBeVisible();
    
    // Check for evolution data if available
    const evolutionData = page.locator('[data-testid="evolution-data"]');
    if (await evolutionData.isVisible()) {
      await expect(page.locator('[data-testid="conservation-score"]')).toBeVisible();
      await expect(page.locator('[data-testid="evolutionary-pressure"]')).toBeVisible();
    }
  });

  test('should clear analysis results', async ({ page }) => {
    // Load example sequence and analyze
    await page.selectOption('[data-testid="example-sequences"]', '0');
    await page.click('[data-testid="analyze-button"]');
    await page.waitForSelector('[data-testid="analysis-complete"]', { timeout: 30000 });
    
    // Clear results
    await page.click('[data-testid="clear-results"]');
    
    // Check that results are cleared
    await expect(page.locator('[data-testid="analysis-summary"]')).not.toBeVisible();
    await expect(page.locator('[data-testid="no-analysis-state"]')).toBeVisible();
  });

  test('should handle batch processing', async ({ page }) => {
    // Navigate to batch tab
    await page.click('[data-testid="tab-batch"]');
    
    // Check batch processing interface
    await expect(page.locator('[data-testid="batch-processor"]')).toBeVisible();
    await expect(page.locator('[data-testid="batch-upload"]')).toBeVisible();
    
    // Test file upload area
    await expect(page.locator('[data-testid="file-upload-area"]')).toBeVisible();
  });

  test('should be responsive on mobile devices', async ({ page, isMobile }) => {
    if (!isMobile) {
      await page.setViewportSize({ width: 375, height: 667 });
    }
    
    // Check that interface is still functional on mobile
    await expect(page.locator('[data-testid="protein-analysis-interface"]')).toBeVisible();
    
    // Check that sequence input panel is accessible
    await expect(page.locator('[data-testid="sequence-input-panel"]')).toBeVisible();
    
    // Check that tabs are accessible (might be in a different layout)
    const tabsVisible = await page.locator('[data-testid="tab-sequence"]').isVisible();
    expect(tabsVisible).toBeTruthy();
  });

  test('should handle network errors gracefully', async ({ page }) => {
    // Intercept analysis requests and simulate failure
    await page.route('**/api/esm3/**', route => {
      route.abort('failed');
    });
    
    // Try to analyze sequence
    await page.selectOption('[data-testid="example-sequences"]', '0');
    await page.click('[data-testid="analyze-button"]');
    
    // Should show error message or fallback to mock data
    await page.waitForTimeout(5000);
    
    const errorMessage = page.locator('[data-testid="analysis-error"]');
    const mockResults = page.locator('[data-testid="analysis-summary"]');
    
    // Should show either error or mock results
    const hasError = await errorMessage.isVisible();
    const hasMockResults = await mockResults.isVisible();
    
    expect(hasError || hasMockResults).toBeTruthy();
  });
});
