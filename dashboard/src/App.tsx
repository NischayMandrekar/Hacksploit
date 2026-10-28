import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AppShell } from './components/AppShell';
import { DashboardHome } from './pages/DashboardHome';
import { ApplicationsPage } from './pages/ApplicationsPage';
import { ApplicationDetailPage } from './pages/ApplicationDetailPage';
import { AnalyticsPage } from './pages/AnalyticsPage';
import { FraudIntelligencePage } from './pages/FraudIntelligencePage';
import { UsersPage } from './pages/UsersPage';
import { ComponentShowcase } from './pages/ComponentShowcase';
import './styles/dashboard.css';

export function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<DashboardHome />} />
          <Route path="/applications" element={<ApplicationsPage />} />
          <Route path="/applications/:id" element={<ApplicationDetailPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
          <Route path="/fraud-intelligence" element={<FraudIntelligencePage />} />
          <Route path="/users" element={<UsersPage />} />
          <Route path="/showcase" element={<ComponentShowcase />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}

export default App;
