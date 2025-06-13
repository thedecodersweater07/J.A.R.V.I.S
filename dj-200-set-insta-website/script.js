// DJ Instagram Guide JavaScript
document.addEventListener('DOMContentLoaded', function() {
    
    // Card hover effects
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-10px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });

    // Hashtag click to copy
    const hashtagLists = document.querySelectorAll('.hashtag-list');
    hashtagLists.forEach(list => {
        list.addEventListener('click', function() {
            const text = this.textContent.trim();
            navigator.clipboard.writeText(text).then(() => {
                showNotification('Hashtags gekopieerd! 🎵');
            });
        });
    });

    // Performance table row highlights
    const tableRows = document.querySelectorAll('.performance-table tbody tr');
    tableRows.forEach(row => {
        row.addEventListener('click', function() {
            const day = this.cells[0].textContent;
            const time = this.cells[2].textContent;
            if (time !== '-') {
                showNotification(`📅 ${day} om ${time} - Perfect timing! 🔥`);
            }
        });
    });

    // Auto-scroll animation for grid background
    let scrollPosition = 0;
    const gridBackground = document.querySelector('.grid-background');
    
    function animateGrid() {
        scrollPosition += 0.5;
        if (scrollPosition >= 50) scrollPosition = 0;
        gridBackground.style.backgroundPosition = `${scrollPosition}px ${scrollPosition}px`;
        requestAnimationFrame(animateGrid);
    }
    animateGrid();

});

// Login/Auth Functions
function validateForm(formType) {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    
    if (!username || !password) {
        showNotification('❌ Vul alle velden in!', 'error');
        return false;
    }
    
    if (password.length < 6) {
        showNotification('❌ Wachtwoord moet minimaal 6 tekens zijn!', 'error');
        return false;
    }
    
    return true;
}

function handleLogin(event) {
    event.preventDefault();
    
    if (!validateForm('login')) return;
    
    const form = event.target;
    const submitBtn = form.querySelector('button[type="submit"]');
    const originalBtnText = submitBtn.textContent;
    
    // Show loading state
    submitBtn.disabled = true;
    submitBtn.textContent = 'Inloggen...';
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const remember = document.getElementById('remember').checked;
    
    // Send to Python backend
    fetch(form.action, {
        method: form.method,
        headers: { 
            'Content-Type': 'application/json' 
        },
        body: JSON.stringify({ 
            username: username, 
            password: password, 
            remember: remember 
        })
    })
    .then(async response => {
        const data = await response.json();
        console.log('Login response:', data);
        
        if (response.ok && data.status === 'success') {
            showNotification('✅ ' + (data.message || 'Login succesvol! 🎧'), 'success');
            // Redirect after successful login
            setTimeout(() => {
                // Get redirect URL from URL parameters or use the one from response
                const urlParams = new URLSearchParams(window.location.search);
                const redirectUrl = urlParams.get('redirect') || data.redirect || '/index.html';
                window.location.href = redirectUrl;
            }, 1500);
        } else {
            const errorMessage = data.message || 'Inloggen mislukt';
            showNotification('❌ ' + errorMessage, 'error');
            throw new Error(errorMessage);
        }
    })
    .catch(error => {
        console.error('Login error:', error);
        if (!error.message.includes('Inloggen mislukt')) {
            showNotification('❌ Verbindingsfout! Probeer het opnieuw.', 'error');
        }
    })
    .finally(() => {
        // Reset button state
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.textContent = originalBtnText;
        }
    });
}

function handleRegister(event) {
    event.preventDefault();
    
    if (!validateForm('register')) return;
    
    const email = document.getElementById('email').value;
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const form = event.target;
    
    // Show loading state
    const submitBtn = form.querySelector('input[type="submit"]');
    const originalBtnText = submitBtn.value;
    submitBtn.value = 'Bezig met registreren...';
    submitBtn.disabled = true;
    
    // Prepare form data as JSON
    const formData = {
        username: username,
        password: password,
        email: email
    };
    
    // Send to Python backend
    fetch(form.action, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData)
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => { throw err; });
        }
        return response.json();
    })
    .then(data => {
        if (data.status === 'success') {
            showNotification('✅ ' + (data.message || 'Registratie succesvol!'), 'success');
            if (data.redirect) {
                setTimeout(() => window.location.href = data.redirect, 1500);
            }
        } else {
            throw new Error(data.message || 'Registratie mislukt');
        }
    })
    .catch(error => {
        showNotification('❌ ' + (error.message || 'Registratie mislukt!'), 'error');
    })
    .finally(() => {
        submitBtn.value = originalBtnText;
        submitBtn.disabled = false;
    });
}

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 2rem;
        background: ${type === 'error' ? '#ff4444' : type === 'success' ? '#00ff88' : '#00ccff'};
        color: #0a0a0a;
        border-radius: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        z-index: 1000;
        animation: slideIn 0.3s ease;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(style);