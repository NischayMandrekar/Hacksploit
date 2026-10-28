import { Search, Bell, Menu, Shield } from 'lucide-react';
import { IconButton } from './IconButton';

interface TopBarProps {
  onMenuToggle?: () => void;
  searchValue?: string;
  onSearchChange?: (val: string) => void;
  onSearchSubmit?: () => void;
}

export function TopBar({
  onMenuToggle,
  searchValue = '',
  onSearchChange,
  onSearchSubmit,
}: TopBarProps) {
  return (
    <header className="dash-topbar">
      <div className="dash-topbar-left">
        <button
          className="dash-menu-toggle"
          onClick={onMenuToggle}
          aria-label="Toggle navigation menu"
        >
          <Menu size={20} />
        </button>

        <form
          className="dash-search-bar"
          onSubmit={(e) => {
            e.preventDefault();
            if (onSearchSubmit) onSearchSubmit();
          }}
        >
          <Search size={15} />
          <input
            type="text"
            placeholder="Global search by applicant name, ID, national ID, or phone..."
            value={searchValue}
            onChange={(e) => onSearchChange?.(e.target.value)}
          />
        </form>
      </div>

      <div className="dash-topbar-right">
        <IconButton title="System Status" aria-label="Security Shield">
          <Shield size={16} />
        </IconButton>

        <div style={{ position: 'relative' }}>
          <IconButton title="Notifications" aria-label="Notifications">
            <Bell size={16} />
          </IconButton>
          <span
            style={{
              position: 'absolute',
              top: 4,
              right: 4,
              width: 7,
              height: 7,
              borderRadius: '50%',
              backgroundColor: 'var(--dash-accent)',
            }}
          />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', marginLeft: '0.5rem' }}>
          <div className="dash-avatar" title="Senior Compliance Officer">
            SC
          </div>
          <div className="dash-topbar-user">
            <span className="dash-topbar-username">Sarah Chen</span>
            <span className="dash-topbar-userrole">Senior Fraud Analyst</span>
          </div>
        </div>
      </div>
    </header>
  );
}

