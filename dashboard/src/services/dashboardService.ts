// Dashboard Service Layer
// Keeps UI components decoupled from data fetching.
// In development, returns typed sample data. When backend is ready,
// replace these calls with fetch('/api/v1/...') or the project's API client.

import type {
  Application,
  DashboardUser,
  AnalyticsSeries,
  SharedIdentifierCluster,
  FraudCluster,
  MetricCardData,
} from '../types/dashboard';
import {
  SAMPLE_APPLICATIONS,
  SAMPLE_USERS,
  SAMPLE_ANALYTICS,
  SAMPLE_SHARED_CLUSTERS,
  SAMPLE_FRAUD_CLUSTERS,
  DASHBOARD_SUMMARY_METRICS,
} from '../data/sampleData';

class DashboardService {
  private applications: Application[] = [...SAMPLE_APPLICATIONS];
  private users: DashboardUser[] = [...SAMPLE_USERS];

  async getSummaryMetrics(): Promise<MetricCardData[]> {
    // Simulated network delay
    await new Promise((r) => setTimeout(r, 80));
    return DASHBOARD_SUMMARY_METRICS;
  }

  async getApplications(): Promise<Application[]> {
    await new Promise((r) => setTimeout(r, 100));
    return [...this.applications];
  }

  async getApplicationById(id: string): Promise<Application | null> {
    await new Promise((r) => setTimeout(r, 80));
    const found = this.applications.find((a) => a.id === id || a.appId === id);
    return found ? { ...found } : null;
  }

  async updateApplicationStatus(id: string, status: Application['status'], note?: string): Promise<Application | null> {
    await new Promise((r) => setTimeout(r, 150));
    const index = this.applications.findIndex((a) => a.id === id || a.appId === id);
    if (index === -1) return null;

    const current = this.applications[index];
    const updated: Application = {
      ...current,
      status,
      updatedAt: new Date().toISOString(),
      history: [
        ...current.history,
        {
          timestamp: new Date().toISOString(),
          action: `Status marked as ${status}`,
          actor: 'Sarah Chen (Analyst)',
          note,
        },
      ],
    };
    this.applications[index] = updated;
    return updated;
  }

  async getUsers(): Promise<DashboardUser[]> {
    await new Promise((r) => setTimeout(r, 80));
    return [...this.users];
  }

  async addUser(user: Omit<DashboardUser, 'id' | 'createdAt' | 'lastLogin'>): Promise<DashboardUser> {
    await new Promise((r) => setTimeout(r, 150));
    const newUser: DashboardUser = {
      ...user,
      id: `usr-${Date.now()}`,
      createdAt: new Date().toISOString(),
      lastLogin: 'Never',
    };
    this.users.unshift(newUser);
    return newUser;
  }

  async deleteUser(id: string): Promise<boolean> {
    await new Promise((r) => setTimeout(r, 100));
    const initial = this.users.length;
    this.users = this.users.filter((u) => u.id !== id);
    return this.users.length < initial;
  }

  async getAnalytics(): Promise<AnalyticsSeries> {
    await new Promise((r) => setTimeout(r, 120));
    return SAMPLE_ANALYTICS;
  }

  async getFraudClusters(): Promise<{ shared: SharedIdentifierCluster[]; rings: FraudCluster[] }> {
    await new Promise((r) => setTimeout(r, 100));
    return {
      shared: SAMPLE_SHARED_CLUSTERS,
      rings: SAMPLE_FRAUD_CLUSTERS,
    };
  }
}

export const dashboardService = new DashboardService();

