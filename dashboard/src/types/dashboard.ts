// Dashboard TypeScript interfaces — VeriYID KYC & Fraud Intelligence Admin Dashboard

export type ApplicationStatus = 'verified' | 'pending' | 'flagged' | 'review_required';
export type DocumentType = 'passport' | 'national_id' | 'drivers_license' | 'residence_permit';
export type UserRole = 'admin' | 'analyst' | 'reviewer' | 'viewer';
export type UserStatus = 'active' | 'inactive' | 'suspended';
export type RiskLevel = 'low' | 'medium' | 'high' | 'critical';

export interface Applicant {
  id: string;
  appId: string;
  fullName: string;
  email: string;
  phone: string;
  dateOfBirth: string;
  nationality: string;
  address: string;
  city: string;
  country: string;
  deviceFingerprint?: string;
  ipAddress?: string;
}

export interface Document {
  type: DocumentType;
  documentNumber: string;
  issuingCountry: string;
  issueDate: string;
  expiryDate: string;
  verified: boolean;
  extractedData?: Record<string, string>;
}

export interface BiometricData {
  selfieUrl?: string;
  livenessScore: number;
  faceMatchScore: number;
  antiSpoofScore: number;
  passed: boolean;
  tamperingStatus?: string;
}

export interface RiskIndicator {
  code: string;
  label: string;
  severity: RiskLevel;
  description: string;
}

export interface Mismatch {
  field: string;
  documentValue: string;
  declaredValue: string;
}

export interface RiskAssessment {
  score: number; // 0-100
  level: RiskLevel;
  indicators: RiskIndicator[];
  mismatches: Mismatch[];
  sharedIdentifiers: string[];
}

export interface HistoryEvent {
  timestamp: string;
  action: string;
  actor: string;
  note?: string;
}

export interface Application {
  id: string;
  appId: string;
  applicant: Applicant;
  status: ApplicationStatus;
  documentType: DocumentType;
  document: Document;
  biometric: BiometricData;
  risk: RiskAssessment;
  history: HistoryEvent[];
  submittedAt: string;
  updatedAt: string;
}

export interface DashboardUser {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  status: UserStatus;
  lastLogin: string;
  createdAt: string;
}

export interface ChartDataPoint {
  label: string;
  value: number;
}

export interface ChartDataPointColored {
  label: string;
  value: number;
  color: string;
}

export interface AnalyticsSeries {
  volumeOverTime: ChartDataPoint[];
  outcomeDistribution: ChartDataPointColored[];
  byDocumentType: ChartDataPoint[];
  fraudFlagsOverTime: ChartDataPoint[];
}

export interface SharedIdentifierCluster {
  id: string;
  identifierType: 'phone' | 'email' | 'device' | 'address';
  identifierValue: string;
  applicationIds: string[];
  firstSeen: string;
  lastSeen: string;
  note: string;
}

export interface FraudCluster {
  id: string;
  clusterLabel: string;
  applicationIds: string[];
  sharedIndicators: string[];
  riskLevel: RiskLevel;
  note: string;
}

export interface MetricCardData {
  label: string;
  value: number | string;
  delta?: number | string;
  deltaLabel?: string;
  iconName?: string;
}

export interface ToastMessage {
  id: string;
  title?: string;
  message: string;
  type?: 'success' | 'error' | 'info';
}

export interface DateRange {
  startDate: string;
  endDate: string;
}
