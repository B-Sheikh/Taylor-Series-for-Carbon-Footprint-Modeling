/**
 * Authentication and Local Storage Manager
 * Mocks user authentication and handles data persistence in localStorage
 */

class AuthManager {
    constructor() {
        this.currentUser = localStorage.getItem('ecocalc_current_user');
        this.users = JSON.parse(localStorage.getItem('ecocalc_users') || '{}');
        this.data = JSON.parse(localStorage.getItem('ecocalc_data') || '{}');
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.updateUI();
    }

    bindEvents() {
        // Modal toggles
        const loginBtn = document.getElementById('loginBtn');
        const signupBtn = document.getElementById('signupBtn');
        const logoutBtn = document.getElementById('logoutBtn');
        const modalClose = document.querySelector('.modal-close');
        const modalOverlay = document.querySelector('.modal-overlay');
        const switchToSignup = document.getElementById('switchToSignup');
        const switchToLogin = document.getElementById('switchToLogin');
        
        // Forms
        const loginFormElement = document.getElementById('loginFormElement');
        const signupFormElement = document.getElementById('signupFormElement');

        // CTA buttons
        const ctaSignupBtn = document.getElementById('ctaSignupBtn');
        
        // Bind modal open events
        if (loginBtn) loginBtn.addEventListener('click', () => this.showLoginModal());
        if (signupBtn) signupBtn.addEventListener('click', () => this.showSignupModal());
        if (ctaSignupBtn) ctaSignupBtn.addEventListener('click', () => this.showSignupModal());
        
        // Bind modal close events
        if (modalClose) modalClose.addEventListener('click', () => this.closeModal());
        if (modalOverlay) modalOverlay.addEventListener('click', () => this.closeModal());
        
        // Form switching
        if (switchToSignup) switchToSignup.addEventListener('click', (e) => {
            e.preventDefault();
            this.showSignupModal();
        });
        
        if (switchToLogin) switchToLogin.addEventListener('click', (e) => {
            e.preventDefault();
            this.showLoginModal();
        });
        
        // Form submissions
        if (loginFormElement) {
            loginFormElement.addEventListener('submit', (e) => {
                e.preventDefault();
                const email = document.getElementById('loginEmail').value;
                const pass = document.getElementById('loginPassword').value;
                this.login(email, pass);
            });
        }
        
        if (signupFormElement) {
            signupFormElement.addEventListener('submit', (e) => {
                e.preventDefault();
                const name = document.getElementById('signupName').value;
                const email = document.getElementById('signupEmail').value;
                const pass = document.getElementById('signupPassword').value;
                const confirm = document.getElementById('signupConfirm').value;
                
                if (pass !== confirm) {
                    alert('Passwords do not match');
                    return;
                }
                
                this.signup(name, email, pass);
            });
        }
        
        if (logoutBtn) logoutBtn.addEventListener('click', () => this.logout());
    }

    showLoginModal() {
        const modal = document.getElementById('authModal');
        const loginForm = document.getElementById('loginForm');
        const signupForm = document.getElementById('signupForm');
        
        if (modal && loginForm && signupForm) {
            modal.classList.remove('hidden');
            loginForm.classList.remove('hidden');
            signupForm.classList.add('hidden');
        }
    }

    showSignupModal() {
        const modal = document.getElementById('authModal');
        const loginForm = document.getElementById('loginForm');
        const signupForm = document.getElementById('signupForm');
        
        if (modal && loginForm && signupForm) {
            modal.classList.remove('hidden');
            signupForm.classList.remove('hidden');
            loginForm.classList.add('hidden');
        }
    }

    closeModal() {
        const modal = document.getElementById('authModal');
        if (modal) modal.classList.add('hidden');
    }

    login(email, password) {
        if (this.users[email] && this.users[email].password === password) {
            this.currentUser = email;
            localStorage.setItem('ecocalc_current_user', email);
            this.closeModal();
            this.updateUI();
            
            // Reload page if on dashboard
            if (window.location.pathname.includes('dashboard.html')) {
                window.location.reload();
            }
        } else {
            alert('Invalid email or password');
        }
    }

    signup(name, email, password) {
        if (this.users[email]) {
            alert('Email already registered');
            return;
        }
        
        this.users[email] = { name, password };
        localStorage.setItem('ecocalc_users', JSON.stringify(this.users));
        
        // Initialize data array for new user
        this.data[email] = [];
        localStorage.setItem('ecocalc_data', JSON.stringify(this.data));
        
        this.login(email, password);
    }

    logout() {
        this.currentUser = null;
        localStorage.removeItem('ecocalc_current_user');
        this.updateUI();
        
        if (window.location.pathname.includes('dashboard.html')) {
            window.location.reload();
        }
    }

    updateUI() {
        const authLinks = document.getElementById('authLinks');
        const userMenu = document.getElementById('userMenu');
        const userName = document.getElementById('userName');
        const saveBtn = document.getElementById('saveBtn');
        
        if (this.currentUser) {
            if (authLinks) authLinks.classList.add('hidden');
            if (userMenu) userMenu.classList.remove('hidden');
            if (userName) userName.textContent = this.users[this.currentUser].name;
            if (saveBtn) saveBtn.classList.remove('hidden');
        } else {
            if (authLinks) authLinks.classList.remove('hidden');
            if (userMenu) userMenu.classList.add('hidden');
            if (saveBtn) saveBtn.classList.add('hidden');
        }
    }

    isLoggedIn() {
        return !!this.currentUser;
    }

    getCurrentUserData() {
        if (!this.currentUser) return [];
        return this.data[this.currentUser] || [];
    }

    saveEntry(entry) {
        if (!this.currentUser) return false;
        
        if (!this.data[this.currentUser]) {
            this.data[this.currentUser] = [];
        }
        
        // Check if entry for this date already exists
        const existingIndex = this.data[this.currentUser].findIndex(e => e.date === entry.date);
        
        if (existingIndex >= 0) {
            this.data[this.currentUser][existingIndex] = entry;
        } else {
            this.data[this.currentUser].push(entry);
        }
        
        // Sort by date
        this.data[this.currentUser].sort((a, b) => new Date(a.date) - new Date(b.date));
        
        localStorage.setItem('ecocalc_data', JSON.stringify(this.data));
        return true;
    }

    deleteEntry(date) {
        if (!this.currentUser) return false;
        
        this.data[this.currentUser] = this.data[this.currentUser].filter(e => e.date !== date);
        localStorage.setItem('ecocalc_data', JSON.stringify(this.data));
        return true;
    }
}

// Initialize AuthManager globally
const authManager = new AuthManager();

// Special handling for the save button on the calculator page
document.addEventListener('DOMContentLoaded', () => {
    const saveBtn = document.getElementById('saveBtn');
    if (saveBtn) {
        saveBtn.addEventListener('click', () => {
            if (!authManager.isLoggedIn()) {
                authManager.showLoginModal();
                return;
            }
            
            // Get calculator UI instance if it exists and trigger a calculation to save
            const today = new Date();
            const currentMonth = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}`;
            
            const inputs = [
                parseFloat(document.getElementById('electricity').value) || 0,
                parseFloat(document.getElementById('fossil_fuel').value) || 0,
                parseFloat(document.getElementById('natural_gas').value) || 0,
                parseFloat(document.getElementById('transportation').value) || 0,
                parseFloat(document.getElementById('industrial').value) || 0,
                parseFloat(document.getElementById('waste').value) || 0,
                parseFloat(document.getElementById('renewable').value) || 0
            ];
            
            // We need the model to calculate emissions, recreating minimal logic here
            const e = [0.5, 2.7, 2.0, 0.12, 1.5, 0.5, -0.3];
            const a = [0.0001, 0.0005, 0.0003, 0.00001, 0.001, 0.0002, 0.00005];
            const B = [
                [0,      0.0001, 0.00005, 0.00002, 0.0003,  0.00001, -0.00005],
                [0.0001, 0,      0.0002,  0.00003, 0.0001,  0.00005, -0.00002],
                [0.00005,0.0002, 0,       0.00001, 0.00015, 0.00002, -0.00001],
                [0.00002,0.00003,0.00001, 0,       0.00008, 0.00001, -0.00003],
                [0.0003, 0.0001, 0.00015, 0.00008, 0,       0.0002,  -0.00004],
                [0.00001,0.00005,0.00002, 0.00001, 0.0002,  0,       0.00001],
                [-0.00005,-0.00002,-0.00001,-0.00003,-0.00004,0.00001,0]
            ];
            
            let linear = 0;
            for (let i = 0; i < 7; i++) linear += e[i] * inputs[i];
            
            let quadratic = 0;
            for (let i = 0; i < 7; i++) quadratic += 0.5 * a[i] * inputs[i] * inputs[i];
            
            let interaction = 0;
            for (let i = 0; i < 7; i++) {
                for (let j = 0; j < 7; j++) {
                    interaction += 0.5 * inputs[i] * B[i][j] * inputs[j];
                }
            }
            
            const totalEmissions = linear + quadratic + interaction;
            
            const entry = {
                date: currentMonth,
                electricity: inputs[0],
                fossilFuel: inputs[1],
                naturalGas: inputs[2],
                transportation: inputs[3],
                industrial: inputs[4],
                waste: inputs[5],
                renewable: inputs[6],
                totalEmissions: totalEmissions,
                timestamp: new Date().toISOString()
            };
            
            if (authManager.saveEntry(entry)) {
                alert('Data saved to dashboard successfully!');
            }
        });
    }
});
