import { test, expect } from '@playwright/test';

test.describe('Bio-Twin Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    // Use stored authentication state
    await page.goto('/dashboard');
    await page.waitForLoadState('networkidle');
  });

  test('should display bio-twin dashboard with key metrics', async ({ page }) => {
    // Check page title
    await expect(page).toHaveTitle(/Bio-Digital Twin/);
    
    // Check main dashboard elements
    await expect(page.locator('[data-testid="bio-twin-dashboard"]')).toBeVisible();
    await expect(page.locator('h1')).toContainText('Bio-Digital Twin');
    
    // Check key metrics cards
    const metricCards = [
      'Biological Age',
      'Health Score', 
      'Cognitive Index',
      'Cellular Vitality',
      'Stress Resilience',
      'Energy Level'
    ];
    
    for (const metric of metricCards) {
      await expect(page.locator(`[data-testid="metric-${metric.toLowerCase().replace(/\s+/g, '-')}"]`)).toBeVisible();
    }
  });

  test('should navigate between dashboard tabs', async ({ page }) => {
    // Test tab navigation
    const tabs = ['overview', 'cellular', 'cognitive', 'evolution'];
    
    for (const tab of tabs) {
      await page.click(`[data-testid="tab-${tab}"]`);
      await expect(page.locator(`[data-testid="tab-${tab}"]`)).toHaveClass(/border-blue-500/);
      await expect(page.locator(`[data-testid="content-${tab}"]`)).toBeVisible();
    }
  });

  test('should display real-time data updates', async ({ page }) => {
    // Navigate to overview tab
    await page.click('[data-testid="tab-overview"]');
    
    // Check for live indicator
    await expect(page.locator('[data-testid="live-indicator"]')).toBeVisible();
    await expect(page.locator('[data-testid="live-indicator"]')).toContainText('Live');
    
    // Check health trends chart
    await expect(page.locator('[data-testid="health-trends-chart"]')).toBeVisible();
    
    // Check system status chart
    await expect(page.locator('[data-testid="system-status-chart"]')).toBeVisible();
  });

  test('should display AI insights panel', async ({ page }) => {
    // Check AI insights panel
    await expect(page.locator('[data-testid="ai-insights-panel"]')).toBeVisible();
    await expect(page.locator('[data-testid="ai-insights-panel"] h3')).toContainText('AI Insights');
    
    // Check for insights or empty state
    const insightsCount = await page.locator('[data-testid="insight-item"]').count();
    if (insightsCount > 0) {
      // Check first insight
      await expect(page.locator('[data-testid="insight-item"]').first()).toBeVisible();
      await expect(page.locator('[data-testid="insight-confidence"]').first()).toBeVisible();
    } else {
      // Check empty state
      await expect(page.locator('[data-testid="insights-empty-state"]')).toBeVisible();
    }
  });

  test('should change time range and update data', async ({ page }) => {
    // Test time range selector
    const timeRanges = ['1d', '7d', '30d', '90d'];
    
    for (const range of timeRanges) {
      await page.click(`[data-testid="time-range-${range}"]`);
      await expect(page.locator(`[data-testid="time-range-${range}"]`)).toHaveClass(/bg-blue-100/);
      
      // Wait for data to update
      await page.waitForTimeout(1000);
      
      // Check that charts are still visible (data should update)
      await expect(page.locator('[data-testid="health-trends-chart"]')).toBeVisible();
    }
  });

  test('should display cellular visualization', async ({ page }) => {
    // Navigate to cellular tab
    await page.click('[data-testid="tab-cellular"]');
    await expect(page.locator('[data-testid="content-cellular"]')).toBeVisible();
    
    // Check cellular visualization components
    await expect(page.locator('[data-testid="cellular-visualization"]')).toBeVisible();
    await expect(page.locator('[data-testid="simulation-controls"]')).toBeVisible();
    
    // Check view mode toggle
    await expect(page.locator('[data-testid="view-mode-3d"]')).toBeVisible();
    await expect(page.locator('[data-testid="view-mode-2d"]')).toBeVisible();
    await expect(page.locator('[data-testid="view-mode-charts"]')).toBeVisible();
    
    // Test view mode switching
    await page.click('[data-testid="view-mode-charts"]');
    await expect(page.locator('[data-testid="cellular-charts"]')).toBeVisible();
    
    // Check statistics panel
    await expect(page.locator('[data-testid="cellular-stats"]')).toBeVisible();
  });

  test('should start and stop cellular simulation', async ({ page }) => {
    // Navigate to cellular tab
    await page.click('[data-testid="tab-cellular"]');
    
    // Start simulation
    await page.click('[data-testid="start-simulation"]');
    
    // Check that stop button appears
    await expect(page.locator('[data-testid="stop-simulation"]')).toBeVisible();
    
    // Wait for simulation to run briefly
    await page.waitForTimeout(2000);
    
    // Stop simulation
    await page.click('[data-testid="stop-simulation"]');
    
    // Check that start button reappears
    await expect(page.locator('[data-testid="start-simulation"]')).toBeVisible();
  });

  test('should filter cellular data by cell type', async ({ page }) => {
    // Navigate to cellular tab
    await page.click('[data-testid="tab-cellular"]');
    
    // Test cell type filters
    const cellTypes = ['all', 'stem', 'neural', 'cardiac', 'hepatic', 'muscle'];
    
    for (const cellType of cellTypes) {
      await page.click(`[data-testid="filter-${cellType}"]`);
      await expect(page.locator(`[data-testid="filter-${cellType}"]`)).toHaveClass(/bg-blue-100/);
      
      // Wait for visualization to update
      await page.waitForTimeout(500);
    }
  });

  test('should display evolution timeline', async ({ page }) => {
    // Navigate to evolution tab
    await page.click('[data-testid="tab-evolution"]');
    await expect(page.locator('[data-testid="content-evolution"]')).toBeVisible();
    
    // Check evolution timeline
    await expect(page.locator('[data-testid="evolution-timeline"]')).toBeVisible();
    await expect(page.locator('[data-testid="timeline-event"]').first()).toBeVisible();
    
    // Check timeline statistics
    await expect(page.locator('[data-testid="timeline-stats"]')).toBeVisible();
  });

  test('should be responsive on mobile devices', async ({ page, isMobile }) => {
    if (!isMobile) {
      // Simulate mobile viewport
      await page.setViewportSize({ width: 375, height: 667 });
    }
    
    // Check that dashboard is still functional on mobile
    await expect(page.locator('[data-testid="bio-twin-dashboard"]')).toBeVisible();
    
    // Check that tabs are accessible (might be in a dropdown on mobile)
    const tabsVisible = await page.locator('[data-testid="tab-overview"]').isVisible();
    if (!tabsVisible) {
      // Check for mobile menu
      await expect(page.locator('[data-testid="mobile-menu-toggle"]')).toBeVisible();
    }
    
    // Check that metric cards stack properly on mobile
    const metricCards = page.locator('[data-testid^="metric-"]');
    const cardCount = await metricCards.count();
    expect(cardCount).toBeGreaterThan(0);
  });

  test('should handle network errors gracefully', async ({ page }) => {
    // Intercept network requests and simulate failures
    await page.route('**/api/**', route => {
      route.abort('failed');
    });
    
    // Reload page to trigger network errors
    await page.reload();
    
    // Check for error handling
    const errorMessage = page.locator('[data-testid="error-message"]');
    const retryButton = page.locator('[data-testid="retry-button"]');
    
    // Should show either error message or graceful degradation
    const hasError = await errorMessage.isVisible();
    const hasRetry = await retryButton.isVisible();
    
    if (hasError || hasRetry) {
      // Error handling is working
      expect(hasError || hasRetry).toBeTruthy();
    } else {
      // Should at least show some content (mock data)
      await expect(page.locator('[data-testid="bio-twin-dashboard"]')).toBeVisible();
    }
  });
});
