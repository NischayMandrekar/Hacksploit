import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Network,
  Share2,
  ShieldAlert,
  Info,
} from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { GlassPanel } from '../components/GlassPanel';
import { LoadingState } from '../components/LoadingState';
import { DataTable, type Column } from '../components/DataTable';
import { dashboardService } from '../services/dashboardService';
import type { SharedIdentifierCluster, FraudCluster } from '../types/dashboard';

export function FraudIntelligencePage() {
  const navigate = useNavigate();
  const [clusters, setClusters] = useState<{ shared: SharedIdentifierCluster[]; rings: FraudCluster[] } | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const res = await dashboardService.getFraudClusters();
        setClusters(res);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading || !clusters) {
    return <LoadingState message="Loading fraud intelligence graph and cluster models..." />;
  }

  const sharedColumns: Column<SharedIdentifierCluster>[] = [
    {
      key: 'identifierType',
      header: 'Identifier Type',
      render: (row) => (
        <span style={{ fontWeight: 700, textTransform: 'capitalize' }}>
          {row.identifierType}
        </span>
      ),
    },
    {
      key: 'identifierValue',
      header: 'Pseudonymized Key / Hash',
      render: (row) => (
        <code style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--dash-text-primary)' }}>
          {row.identifierValue}
        </code>
      ),
    },
    {
      key: 'applicationIds',
      header: 'Associated Applications',
      render: (row) => (
        <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
          {row.applicationIds.map((id) => (
            <button
              key={id}
              onClick={() => navigate(`/applications`)}
              style={{
                fontSize: '0.8rem',
                padding: '0.15rem 0.5rem',
                borderRadius: '4px',
                border: '1px solid rgba(18,63,61,0.2)',
                background: 'rgba(255,255,255,0.7)',
                color: 'var(--dash-text-primary)',
                cursor: 'pointer',
                fontWeight: 600,
              }}
            >
              {id}
            </button>
          ))}
        </div>
      ),
    },
    {
      key: 'activity',
      header: 'Observed Range',
      render: (row) => (
        <span style={{ fontSize: '0.85rem', color: 'var(--dash-text-secondary)' }}>
          {row.firstSeen} → {row.lastSeen}
        </span>
      ),
    },
    {
      key: 'note',
      header: 'Intelligence Finding',
      render: (row) => (
        <span style={{ fontSize: '0.88rem', color: 'var(--dash-text-secondary)' }}>
          {row.note}
        </span>
      ),
    },
  ];

  return (
    <>
      <PageHeader
        title="Fraud Intelligence & Syndicate Detection"
        subtitle="Graph-based cross-application correlation, repeated identifier detection, and synthetic cluster screening."
      />

      {/* Critical Methodological Notice */}
      <div className="dash-fraud-banner">
        <Info size={20} />
        <div>
          <strong>Evidence vs Correlation Notice:</strong> An observed shared identifier (such as a device or subnet) indicates an identity relation, not definitive fraud. Full confirmation requires forensic proof of tampering or statutory credential invalidity.
        </div>
      </div>

      {/* Interactive Entity Relationship Graph Visualizer */}
      <GlassPanel style={{ marginBottom: '1.75rem' }}>
        <div className="dash-section-heading">
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Network size={18} /> Identity Ego-Network & Relational Graph Visualizer
          </span>
          <span style={{ fontSize: '0.85rem', color: 'var(--dash-text-muted)', fontWeight: 600 }}>
            Bounded 2-Hop Ego Projections
          </span>
        </div>

        <div className="dash-panel-body">
          <div className="dash-graph-container">
            <svg className="dash-graph-svg" viewBox="0 0 800 360" role="img" aria-label="Identity relational graph network">
              {/* Edges */}
              <line x1="400" y1="180" x2="220" y2="100" stroke="rgba(18,63,61,0.3)" strokeWidth="2" strokeDasharray="4 4" />
              <line x1="400" y1="180" x2="260" y2="280" stroke="rgba(18,63,61,0.3)" strokeWidth="2" strokeDasharray="4 4" />
              <line x1="400" y1="180" x2="580" y2="110" stroke="rgba(18,63,61,0.3)" strokeWidth="2" strokeDasharray="4 4" />
              <line x1="400" y1="180" x2="550" y2="270" stroke="rgba(18,63,61,0.3)" strokeWidth="2" strokeDasharray="4 4" />

              <line x1="220" y1="100" x2="110" y2="70" stroke="rgba(18,63,61,0.25)" strokeWidth="1.5" />
              <line x1="220" y1="100" x2="130" y2="160" stroke="rgba(18,63,61,0.25)" strokeWidth="1.5" />
              <line x1="260" y1="280" x2="150" y2="290" stroke="rgba(18,63,61,0.25)" strokeWidth="1.5" />
              <line x1="580" y1="110" x2="690" y2="90" stroke="rgba(18,63,61,0.25)" strokeWidth="1.5" />
              <line x1="550" y1="270" x2="680" y2="280" stroke="rgba(18,63,61,0.25)" strokeWidth="1.5" />

              {/* Central Hub Node (Shared Device) */}
              <g
                className="dash-graph-node"
                transform="translate(400, 180)"
                onClick={() => setSelectedNode('dev-SYNTH-994 (Shared Device Hub)')}
              >
                <circle r="36" fill="var(--dash-accent)" />
                <circle r="44" fill="none" stroke="rgba(18,63,61,0.25)" strokeWidth="2" />
                <text textAnchor="middle" dy="4" fill="#ffffff" fontSize="11" fontWeight="700">
                  Shared Device
                </text>
              </g>

              {/* Identity Nodes */}
              <g
                className="dash-graph-node"
                transform="translate(220, 100)"
                onClick={() => setSelectedNode('KYC-2026-003: Rajesh K. Varma (Flagged GAN Face)')}
              >
                <circle r="26" fill="rgba(239, 68, 68, 0.85)" />
                <text textAnchor="middle" dy="4" fill="#111111" fontSize="10" fontWeight="700">
                  KYC-003
                </text>
              </g>

              <g
                className="dash-graph-node"
                transform="translate(260, 280)"
                onClick={() => setSelectedNode('KYC-2026-048: Synthetic Identity Entity')}
              >
                <circle r="24" fill="rgba(239, 68, 68, 0.85)" />
                <text textAnchor="middle" dy="4" fill="#111111" fontSize="10" fontWeight="700">
                  KYC-048
                </text>
              </g>

              <g
                className="dash-graph-node"
                transform="translate(580, 110)"
                onClick={() => setSelectedNode('KYC-2026-052: Linked Ingestion Node')}
              >
                <circle r="24" fill="rgba(245, 158, 11, 0.85)" />
                <text textAnchor="middle" dy="4" fill="#111111" fontSize="10" fontWeight="700">
                  KYC-052
                </text>
              </g>

              <g
                className="dash-graph-node"
                transform="translate(550, 270)"
                onClick={() => setSelectedNode('KYC-2026-077: Under Investigation')}
              >
                <circle r="24" fill="rgba(245, 158, 11, 0.85)" />
                <text textAnchor="middle" dy="4" fill="#111111" fontSize="10" fontWeight="700">
                  KYC-077
                </text>
              </g>

              {/* Peripheral Attribute Nodes */}
              <g className="dash-graph-node" transform="translate(110, 70)">
                <circle r="16" fill="rgba(255,255,255,0.85)" stroke="rgba(0,0,0,0.15)" />
                <text textAnchor="middle" dy="4" fill="#111111" fontSize="8" fontWeight="600">IP-103</text>
              </g>
              <g className="dash-graph-node" transform="translate(130, 160)">
                <circle r="16" fill="rgba(255,255,255,0.85)" stroke="rgba(0,0,0,0.15)" />
                <text textAnchor="middle" dy="4" fill="#111111" fontSize="8" fontWeight="600">TempMail</text>
              </g>
              <g className="dash-graph-node" transform="translate(150, 290)">
                <circle r="16" fill="rgba(255,255,255,0.85)" stroke="rgba(0,0,0,0.15)" />
                <text textAnchor="middle" dy="4" fill="#111111" fontSize="8" fontWeight="600">SIM#2</text>
              </g>
              <g className="dash-graph-node" transform="translate(690, 90)">
                <circle r="16" fill="rgba(255,255,255,0.85)" stroke="rgba(0,0,0,0.15)" />
                <text textAnchor="middle" dy="4" fill="#111111" fontSize="8" fontWeight="600">Subnet</text>
              </g>
              <g className="dash-graph-node" transform="translate(680, 280)">
                <circle r="16" fill="rgba(255,255,255,0.85)" stroke="rgba(0,0,0,0.15)" />
                <text textAnchor="middle" dy="4" fill="#111111" fontSize="8" fontWeight="600">MAC</text>
              </g>
            </svg>
          </div>

          {selectedNode && (
            <div style={{ marginTop: '0.85rem', padding: '0.75rem 1rem', background: 'rgba(255,255,255,0.6)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.8)' }}>
              <strong>Selected Entity:</strong> {selectedNode}
            </div>
          )}
        </div>
      </GlassPanel>

      {/* Suspicious Syndicate Clusters */}
      <h2 style={{ fontSize: '1.25rem', fontWeight: 800, marginBottom: '1rem', color: 'var(--dash-text-primary)' }}>
        Identified Fraud Clusters & Syndicate Rings
      </h2>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.25rem', marginBottom: '2rem' }}>
        {clusters.rings.map((ring) => (
          <GlassPanel key={ring.id}>
            <div className="dash-section-heading">
              <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <ShieldAlert size={18} />
                <span>{ring.clusterLabel}</span>
              </span>
              <span
                style={{
                  fontSize: '0.78rem',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  padding: '0.2rem 0.55rem',
                  borderRadius: '4px',
                  background: 'rgba(254, 226, 226, 0.9)',
                }}
              >
                {ring.riskLevel}
              </span>
            </div>
            <div className="dash-panel-body">
              <p style={{ fontSize: '0.92rem', color: 'var(--dash-text-secondary)', marginBottom: '1rem' }}>
                {ring.note}
              </p>

              <div style={{ marginBottom: '1rem' }}>
                <span className="dash-detail-label">Shared Cluster Signals:</span>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginTop: '0.35rem' }}>
                  {ring.sharedIndicators.map((sig, i) => (
                    <span
                      key={i}
                      style={{
                        fontSize: '0.82rem',
                        padding: '0.2rem 0.6rem',
                        borderRadius: '4px',
                        background: 'rgba(255,255,255,0.6)',
                        border: '1px solid rgba(0,0,0,0.08)',
                        fontWeight: 600,
                      }}
                    >
                      {sig}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <span className="dash-detail-label">Correlated Applications ({ring.applicationIds.length}):</span>
                <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', marginTop: '0.35rem' }}>
                  {ring.applicationIds.map((id) => (
                    <button
                      key={id}
                      onClick={() => navigate('/applications')}
                      style={{
                        padding: '0.25rem 0.6rem',
                        borderRadius: '4px',
                        border: '1px solid rgba(18,63,61,0.25)',
                        background: 'var(--dash-accent-light)',
                        color: 'var(--dash-text-primary)',
                        fontSize: '0.82rem',
                        fontWeight: 700,
                        cursor: 'pointer',
                      }}
                    >
                      {id}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </GlassPanel>
        ))}
      </div>

      {/* Shared Identifiers Table */}
      <GlassPanel>
        <div className="dash-section-heading">
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Share2 size={18} /> Cross-Application Shared Identifiers
          </span>
        </div>

        <DataTable
          columns={sharedColumns}
          rows={clusters.shared}
          rowKey={(r) => r.id}
        />
      </GlassPanel>
    </>
  );
}
