/**
 * Navigation Bar Component
 * Shows user info and logout button
 */
import React from 'react';
import { useAuth } from '../AuthContext';

function Navbar() {
  const { user, logout } = useAuth();

  return (
    <nav className="navbar">
      <div className="navbar-logo">Curiosity</div>
      
      {user && (
        <div className="navbar-user">
          <span className="user-name">👤 {user.name}</span>
          <button className="btn-logout" onClick={logout}>
            Logout
          </button>
        </div>
      )}
    </nav>
  );
}

export default Navbar;
