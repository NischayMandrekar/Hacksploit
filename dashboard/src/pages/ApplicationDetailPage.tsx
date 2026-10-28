import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Shield,
  FileText,
  User,
  History,
} from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { GlassPanel } from '../components/GlassPanel';
import { StatusBadge } from '../components/StatusBadge';
import { Tabs } from '../components/Tabs';
import { SecondaryButton } from '../components/SecondaryButton';
import { ConfirmationModal } from '../components/ConfirmationModal';
import { ToastNotification } from '../components/ToastNotification';
import { LoadingState } from '../components/LoadingState';
import { ErrorState } from '../components/ErrorState';
import { dashboardService } from '../services/dashboardService';
import type { Application, ToastMessage } from '../types/dashboard';

export function ApplicationDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [application, setApplication] = useState<Application | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  // Action modals
  const [modalType, setModalType] = useState<'verify' | 'flag' | 'review' | null>(null);
  const [actionLoading, setActionLoading] = useState(false);
  const [actionNote, setActionNote] = useState('');
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  useEffect(() => {
    async function loadApp() {
      if (!id) return;
      setLoading(true);
      try {
        const app = await dashboardService.getApplicationById(id);
        setApplication(app);
      } finally {
        setLoading(false);
      }
    }
    loadApp();
  }, [id]);

  const handleAction = async () => {
    if (!application || !modalType) return;
    setActionLoading(true);
    try {
      const newStatus =
        modalType === 'verify' ? 'verified' : modalType === 'flag' ? 'flagged' : 'review_required';
      const updated = await dashboardService.updateApplicationStatus(
        application.id,
        newStatus,
        actionNote || `Manually marked as ${newStatus}`
      );
      if (updated) {
        setApplication(updated);
        setToasts((prev) => [
          ...prev,
          {
            id: `toast-${Date.now()}`,
            type: 'success',
            title: 'Status Updated',
            message: `Application ${application.appId} successfully marked as ${newStatus}.`,
          },
        ]);
      }
      setModalType(null);
      setActionNote('');
    } catch {
      setToasts((prev) => [
        ...prev,
        {
          id: `toast-${Date.now()}`,
          type: 'error',
          title: 'Action Failed',
          message: 'Could not update status. Please try again.',
        },
      ]);
    } finally {
      setActionLoading(false);
    }
  };

  if (loading) {
    return <LoadingState message="Loading application details..." />;
  }

  if (!application) {
    return (
      <ErrorState
        title="Application Not Found"
        message={`No KYC application found with ID "${id}".`}
        action={
          <SecondaryButton onClick={() => navigate('/applications')}>
            Back to Applications
          </SecondaryButton>
        }
      />
    );
  }

  const tabs = [
    { id: 'overview', label: 'Overview & Profile' },
    { id: 'documents', label: 'Document Forensics' },
    { id: 'biometrics', label: 'Biometrics & Liveness' },
    { id: 'risk', label: 'Risk & Fraud Signals', count: application.risk.indicators.length },
    { id: 'history', label: 'Audit Trail', count: application.history.length },
  ];

  const riskCircleClass =
    application.risk.level === 'critical'
      ? 'dash-risk-circle--critical'
      : application.risk.level === 'high'
      ? 'dash-risk-circle--high'
      : application.risk.level === 'medium'
      ? 'dash-risk-circle--medium'
      : 'dash-risk-circle--low';

  return (
    <>
      <button
        onClick={() => navigate('/applications')}
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.4rem',
          background: 'none',
          border: 'none',
          color: 'var(--dash-text-secondary)',
          fontWeight: 700,
          fontSize: '0.9rem',
          cursor: 'pointer',
          marginBottom: '1rem',
        }}
      >
        <ArrowLeft size={16} /> Back to Applications
      </button>

      <PageHeader
        title={application.applicant.fullName}
        subtitle={`Application ID: ${application.appId} • Submitted ${new Date(
          application.submittedAt
        ).toLocaleDateString()}`}
        actions={
          <div style={{ display: 'flex', gap: '0.65rem' }}>
            <StatusBadge status={application.status} />

            <SecondaryButton
              icon={<CheckCircle2 size={15} />}
              onClick={() => setModalType('verify')}
            >
              Verify
            </SecondaryButton>

            <SecondaryButton
              icon={<Clock size={15} />}
              onClick={() => setModalType('review')}
            >
              Request Info
            </SecondaryButton>

            <button
              className="dash-btn-danger"
              style={{ minHeight: 40, padding: '0.5rem 1.1rem', fontSize: '0.9rem' }}
              onClick={() => setModalType('flag')}
            >
              <AlertTriangle size={15} /> Flag Case
            </button>
          </div>
        }
      />

      {/* Tabs navigation */}
      <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab} />

      {/* TAB 1: OVERVIEW */}
      {activeTab === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>
          <GlassPanel>
            <div className="dash-section-heading">
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <User size={18} /> Applicant Demographics
              </span>
            </div>
            <div className="dash-panel-body">
              <div className="dash-detail-grid">
                <div className="dash-detail-field">
                  <span className="dash-detail-label">Full Legal Name</span>
                  <span className="dash-detail-value">{application.applicant.fullName}</span>
                </div>
                <div className="dash-detail-field">
                  <span className="dash-detail-label">Date of Birth</span>
                  <span className="dash-detail-value">{application.applicant.dateOfBirth}</span>
                </div>
                <div className="dash-detail-field">
                  <span className="dash-detail-label">Email Address</span>
                  <span className="dash-detail-value">{application.applicant.email}</span>
                </div>
                <div className="dash-detail-field">
                  <span className="dash-detail-label">Phone Number</span>
                  <span className="dash-detail-value">{application.applicant.phone}</span>
                </div>
                <div className="dash-detail-field">
                  <span className="dash-detail-label">Nationality</span>
                  <span className="dash-detail-value">{application.applicant.nationality}</span>
                </div>
                <div className="dash-detail-field">
                  <span className="dash-detail-label">Address</span>
                  <span className="dash-detail-value">
                    {application.applicant.address}, {application.applicant.city}, {application.applicant.country}
                  </span>
                </div>
              </div>
            </div>
          </GlassPanel>

          <GlassPanel>
            <div className="dash-section-heading">
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Shield size={18} /> Argus Risk Score
              </span>
            </div>
            <div className="dash-panel-body">
              <div className="dash-risk-gauge">
                <div className={`dash-risk-circle ${riskCircleClass}`}>
                  <span>{application.risk.score}</span>
                  <span className="dash-risk-circle-label">Score</span>
                </div>
                <div>
                  <div className="dash-risk-level-text">
                    Tier: {application.risk.level.toUpperCase()}
                  </div>
                  <p style={{ fontSize: '0.88rem', color: 'var(--dash-text-secondary)', marginTop: '0.35rem' }}>
                    Calculated by Project Argus fusion model combining biometric liveness, forensic tamper analysis, and relational identity graph density.
                  </p>
                </div>
              </div>

              {application.risk.sharedIdentifiers.length > 0 && (
                <div style={{ marginTop: '1rem', borderTop: '1px solid rgba(255,255,255,0.5)', paddingTop: '0.75rem' }}>
                  <span className="dash-detail-label">Shared Cluster Identifiers:</span>
                  <ul style={{ listStyleType: 'disc', paddingLeft: '1.25rem', marginTop: '0.4rem', fontSize: '0.9rem' }}>
                    {application.risk.sharedIdentifiers.map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </GlassPanel>
        </div>
      )}

      {/* TAB 2: DOCUMENT FORENSICS */}
      {activeTab === 'documents' && (
        <GlassPanel>
          <div className="dash-section-heading">
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <FileText size={18} /> Document Forensic Analysis
            </span>
            <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>
              Status: {application.document.verified ? 'Verified' : 'Review Required'}
            </span>
          </div>
          <div className="dash-panel-body">
            <div className="dash-detail-grid" style={{ marginBottom: '1.5rem' }}>
              <div className="dash-detail-field">
                <span className="dash-detail-label">Document Type</span>
                <span className="dash-detail-value" style={{ textTransform: 'capitalize' }}>
                  {application.document.type.replace('_', ' ')}
                </span>
              </div>
              <div className="dash-detail-field">
                <span className="dash-detail-label">Document Number</span>
                <span className="dash-detail-value">{application.document.documentNumber}</span>
              </div>
              <div className="dash-detail-field">
                <span className="dash-detail-label">Issuing Country</span>
                <span className="dash-detail-value">{application.document.issuingCountry}</span>
              </div>
              <div className="dash-detail-field">
                <span className="dash-detail-label">Validity Period</span>
                <span className="dash-detail-value">
                  {application.document.issueDate} to {application.document.expiryDate}
                </span>
              </div>
            </div>

            <div style={{ borderTop: '1px solid rgba(255,255,255,0.5)', paddingTop: '1.25rem' }}>
              <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.75rem' }}>
                Extracted OCR Field Consistency Check
              </h3>
              {application.document.extractedData ? (
                <div style={{ background: 'rgba(255,255,255,0.45)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.65)' }}>
                  <pre style={{ fontFamily: 'monospace', fontSize: '0.88rem', color: 'var(--dash-text-primary)' }}>
                    {JSON.stringify(application.document.extractedData, null, 2)}
                  </pre>
                </div>
              ) : (
                <p style={{ color: 'var(--dash-text-muted)', fontSize: '0.9rem' }}>No OCR extracted fields stored.</p>
              )}
            </div>
          </div>
        </GlassPanel>
      )}

      {/* TAB 3: BIOMETRICS & LIVENESS */}
      {activeTab === 'biometrics' && (
        <GlassPanel>
          <div className="dash-section-heading">
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Shield size={18} /> Biometric Liveness & Anti-Spoofing Verification
            </span>
          </div>
          <div className="dash-panel-body">
            <div className="dash-detail-grid" style={{ marginBottom: '1.5rem' }}>
              <div className="dash-detail-field">
                <span className="dash-detail-label">Liveness Confidence</span>
                <span className="dash-detail-value">
                  {Math.round(application.biometric.livenessScore * 100)}%
                </span>
              </div>
              <div className="dash-detail-field">
                <span className="dash-detail-label">Face Match (Selfie vs Document)</span>
                <span className="dash-detail-value">
                  {Math.round(application.biometric.faceMatchScore * 100)}%
                </span>
              </div>
              <div className="dash-detail-field">
                <span className="dash-detail-label">Anti-Spoofing Clean Score</span>
                <span className="dash-detail-value">
                  {Math.round(application.biometric.antiSpoofScore * 100)}%
                </span>
              </div>
              <div className="dash-detail-field">
                <span className="dash-detail-label">Overall Biometric Gate</span>
                <span className="dash-detail-value">
                  {application.biometric.passed ? 'PASSED (Genuine)' : 'FAILED / SUSPICIOUS'}
                </span>
              </div>
            </div>

            <div style={{ borderTop: '1px solid rgba(255,255,255,0.5)', paddingTop: '1rem' }}>
              <span className="dash-detail-label">Tamper Assessment:</span>
              <p style={{ fontSize: '0.95rem', fontWeight: 600, marginTop: '0.25rem' }}>
                {application.biometric.tamperingStatus || 'No tampering detected during video challenge.'}
              </p>
            </div>
          </div>
        </GlassPanel>
      )}

      {/* TAB 4: RISK & FRAUD SIGNALS */}
      {activeTab === 'risk' && (
        <GlassPanel>
          <div className="dash-section-heading">
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <AlertTriangle size={18} /> Detected Risk Factors & Evidence Attributions
            </span>
          </div>
          <div className="dash-panel-body">
            {application.risk.indicators.length === 0 ? (
              <p style={{ color: 'var(--dash-text-secondary)', fontSize: '0.95rem' }}>
                No suspicious risk indicators flagged for this identity.
              </p>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                {application.risk.indicators.map((ind) => (
                  <div
                    key={ind.code}
                    style={{
                      padding: '1rem',
                      borderRadius: '8px',
                      background: 'rgba(255, 255, 255, 0.45)',
                      border: '1px solid rgba(255, 255, 255, 0.65)',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.35rem' }}>
                      <strong style={{ fontSize: '1rem' }}>{ind.label}</strong>
                      <span
                        style={{
                          fontSize: '0.75rem',
                          fontWeight: 700,
                          textTransform: 'uppercase',
                          padding: '0.15rem 0.5rem',
                          borderRadius: '4px',
                          background:
                            ind.severity === 'critical'
                              ? 'rgba(254, 226, 226, 0.9)'
                              : ind.severity === 'high'
                              ? 'rgba(254, 226, 226, 0.7)'
                              : 'rgba(254, 243, 199, 0.8)',
                        }}
                      >
                        {ind.severity}
                      </span>
                    </div>
                    <p style={{ fontSize: '0.9rem', color: 'var(--dash-text-secondary)' }}>
                      {ind.description}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </GlassPanel>
      )}

      {/* TAB 5: AUDIT TRAIL */}
      {activeTab === 'history' && (
        <GlassPanel>
          <div className="dash-section-heading">
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <History size={18} /> Complete Decision History & Audit Log
            </span>
          </div>
          <div className="dash-panel-body">
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {application.history.map((event, idx) => (
                <div key={idx} style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
                  <div
                    style={{
                      width: 10,
                      height: 10,
                      borderRadius: '50%',
                      background: 'var(--dash-accent)',
                      marginTop: 6,
                      flexShrink: 0,
                    }}
                  />
                  <div>
                    <div style={{ fontWeight: 700, fontSize: '0.95rem' }}>{event.action}</div>
                    <div style={{ fontSize: '0.82rem', color: 'var(--dash-text-muted)' }}>
                      {new Date(event.timestamp).toLocaleString()} • {event.actor}
                    </div>
                    {event.note && (
                      <p style={{ fontSize: '0.88rem', marginTop: '0.2rem', color: 'var(--dash-text-secondary)' }}>
                        Note: {event.note}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </GlassPanel>
      )}

      {/* Confirmation Modal */}
      {modalType && (
        <ConfirmationModal
          title={
            modalType === 'verify'
              ? 'Approve and Verify Application'
              : modalType === 'flag'
              ? 'Flag for Fraud Investigation'
              : 'Request Additional Information'
          }
          confirmLabel={
            modalType === 'verify' ? 'Confirm Approval' : modalType === 'flag' ? 'Confirm Flag' : 'Send Request'
          }
          isDanger={modalType === 'flag'}
          loading={actionLoading}
          onConfirm={handleAction}
          onCancel={() => setModalType(null)}
        >
          <p style={{ marginBottom: '1rem' }}>
            {modalType === 'verify'
              ? 'This action marks the applicant identity as genuine and issues verification credentials.'
              : modalType === 'flag'
              ? 'This action escalates the identity into the fraud intelligence review queue for syndicate correlation.'
              : 'The applicant will receive a secure challenge link to provide clarified or higher resolution documentation.'}
          </p>

          <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.35rem' }}>
            Internal Compliance Note (Optional):
          </label>
          <textarea
            className="dash-form-input"
            rows={3}
            value={actionNote}
            onChange={(e) => setActionNote(e.target.value)}
            placeholder="Add reasoning for this audit record..."
            style={{ width: '100%', resize: 'vertical' }}
          />
        </ConfirmationModal>
      )}

      {/* Toast Notifications */}
      <ToastNotification
        toasts={toasts}
        onDismiss={(id) => setToasts((prev) => prev.filter((t) => t.id !== id))}
      />
    </>
  );
}
