document.addEventListener('DOMContentLoaded', function() {
    
    // Element selections
    const portfolio = document.getElementById('portfolio');
    const primaryColor = document.getElementById('primaryColor');
    const secondaryColor = document.getElementById('secondaryColor');
    const fontSize = document.getElementById('fontSize');
    const fontSizeValue = document.getElementById('fontSizeValue');
    const fontFamily = document.getElementById('fontFamily');
    const highContrast = document.getElementById('highContrast');
    const focusIndicator = document.getElementById('focusIndicator');
    const exportBtn = document.getElementById('exportBtn');
    const addBtns = document.querySelectorAll('.add-btn');

    let componentCount = 0;

    // Component templates
    const components = {
        header: function() {
            return `
                <header class="portfolio-header">
                    <div class="component-controls">
                        <button class="delete-btn">Delete</button>
                    </div>
                    <h1 contenteditable="true">Your Name</h1>
                    <p contenteditable="true">Your Title / Role</p>
                </header>
            `;
        },
        about: function() {
            componentCount++;
            return `
                <section class="about-section">
                    <div class="component-controls">
                        <button class="delete-btn">Delete</button>
                    </div>
                    <h2>About Me</h2>
                    <p contenteditable="true">Write about yourself, your skills, and your passions. Make it personal and engaging!</p>
                </section>
            `;
        },
        project: function() {
            componentCount++;
            return `
                <article class="project-card">
                    <div class="component-controls">
                        <button class="delete-btn">Delete</button>
                    </div>
                    <h3 contenteditable="true">Project Title</h3>
                    <p contenteditable="true">Describe your project, the technologies used, and the problem it solves.</p>
                </article>
            `;
        },
        contact: function() {
            componentCount++;
            return `
                <section class="contact-section">
                    <div class="component-controls">
                        <button class="delete-btn">Delete</button>
                    </div>
                    <h2>Get In Touch</h2>
                    <p contenteditable="true">Email: your@email.com</p>
                    <p contenteditable="true">LinkedIn: linkedin.com/in/yourprofile</p>
                </section>
            `;
        }
    };

    // Add component functionality
    addBtns.forEach(function(btn) {
        btn.addEventListener('click', function() {
            const componentType = btn.getAttribute('data-component');
            addComponent(componentType);
        });
    });

    function addComponent(type) {
        // Remove placeholder if exists
        const placeholder = portfolio.querySelector('.placeholder');
        if (placeholder) {
            placeholder.remove();
        }
        
        // Get HTML from templates
        const componentHTML = components[type]();
        
        // Add to portfolio
        portfolio.insertAdjacentHTML('beforeend', componentHTML);
        
        // Attach delete event
        const newComponent = portfolio.lastElementChild;
        const deleteBtn = newComponent.querySelector('.delete-btn');
        
        if (deleteBtn) {
            deleteBtn.addEventListener('click', function() {
                newComponent.remove();
                
                // Add placeholder back if empty
                if (portfolio.children.length === 0) {
                    portfolio.innerHTML = '<p class="placeholder">Add components to build your portfolio</p>';
                }
            });
        }
    }

    // Theme controls
    primaryColor.addEventListener('input', function(e) {
        portfolio.style.setProperty('--primary-color', e.target.value);
    });

    secondaryColor.addEventListener('input', function(e) {
        portfolio.style.setProperty('--secondary-color', e.target.value);
    });

    fontSize.addEventListener('input', function(e) {
        portfolio.style.fontSize = e.target.value + 'px';
        fontSizeValue.textContent = e.target.value + 'px';
    });

    fontFamily.addEventListener('change', function(e) {
        portfolio.style.fontFamily = e.target.value;
    });

    // Accessibility controls
    highContrast.addEventListener('change', function(e) {
        portfolio.classList.toggle('high-contrast', e.target.checked);
    });

    focusIndicator.addEventListener('change', function(e) {
        portfolio.classList.toggle('focus-mode', e.target.checked);
    });

    // Export functionality
    exportBtn.addEventListener('click', function() {
        const primaryVal = primaryColor.value;
        const secondaryVal = secondaryColor.value;
        const fontVal = fontSize.value + 'px';
        const familyVal = fontFamily.value;
        
        const styles = `
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { font-family: ${familyVal}; font-size: ${fontVal}; padding: 40px; background: #f5f5f5; }
                .portfolio-header { text-align: center; padding: 60px 20px; background: linear-gradient(135deg, ${primaryVal}, ${secondaryVal}); color: white; border-radius: 8px; margin-bottom: 30px; }
                .portfolio-header h1 { font-size: 48px; margin-bottom: 10px; }
                .about-section { padding: 40px; margin: 30px 0; background: white; border-left: 5px solid ${primaryVal}; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                .project-card { padding: 30px; margin: 20px 0; background: white; border: 2px solid #e0e0e0; border-radius: 8px; }
                .contact-section { padding: 40px; background: ${secondaryVal}; color: white; border-radius: 8px; margin-top: 30px; text-align: center; }
                .component-controls { display: none; }
            </style>
        `;
        
        const portfolioClone = portfolio.cloneNode(true);
        portfolioClone.querySelectorAll('.component-controls').forEach(el => el.remove());
        portfolioClone.querySelectorAll('[contenteditable]').forEach(el => el.removeAttribute('contenteditable'));
        
        const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Portfolio</title>
    ${styles}
</head>
<body>
    ${portfolioClone.innerHTML}
</body>
</html>
        `;
        
        const blob = new Blob([html], { type: 'text/html' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'my-portfolio.html';
        link.click();
    });

    // Initialize focus mode
    portfolio.classList.add('focus-mode');
});