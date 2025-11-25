# Contributing to YAGC V16f-η

Thank you for your interest in contributing to the YAGC project! This document provides guidelines for contributing to the V16f-η breathing function implementation.

## 🌟 Ways to Contribute

### 1. Independent Verification
- Run the code with default parameters
- Verify that results match published values
- Report any discrepancies via GitHub Issues

### 2. Parameter Exploration
- Test different parameter regimes
- Document interesting behaviors
- Share phase diagrams and visualizations

### 3. Theoretical Extensions
- Connect V16f-η to other cosmological models
- Propose new experiments
- Suggest alternative interpretations

### 4. Code Improvements
- Performance optimizations
- Better documentation
- Additional tests
- Bug fixes

### 5. Visualization
- Create animations of breathing dynamics
- Interactive Jupyter widgets
- Publication-quality figures

## 📋 Contribution Process

### Before You Start

1. **Check existing issues**: See if someone is already working on it
2. **Open a discussion**: For major changes, open an issue first
3. **Read the theory**: Familiarize yourself with V28 Appendix X

### Making Changes

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/v16f-eta.git
   cd v16f-eta
   ```

2. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Write clear, commented code
   - Follow PEP 8 style guidelines
   - Add tests for new features
   - Update documentation

4. **Test your changes**
   ```bash
   pytest tests/
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add: Brief description of changes"
   ```

6. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## 🧪 Testing Requirements

All contributions must include appropriate tests:

### For New Features
```python
# tests/test_your_feature.py
import pytest
from cosmos_v16f import CosmosV16f

def test_your_feature():
    cosmos = CosmosV16f()
    result = cosmos.your_new_method()
    assert result > 0  # Your assertion here
```

### For Bug Fixes
Include a test that would have caught the bug.

## 📝 Documentation Standards

### Code Documentation
```python
def new_function(param1, param2):
    """
    Brief description.
    
    Detailed explanation of what this function does
    and its connection to YAGC theory.
    
    Parameters
    ----------
    param1 : float
        Description of param1
    param2 : int
        Description of param2
        
    Returns
    -------
    result : float
        Description of return value
        
    References
    ----------
    V28 Appendix X, Section 1.2
    """
    pass
```

### Theoretical Context
When adding features, explain their connection to V16→V17R→V18R→V26-28 theory.

## 🔬 Scientific Rigor

### Reproducibility
- Set random seeds explicitly
- Document all parameters
- Include version numbers of dependencies

### Validation
- Compare with published results
- Cross-check with V18R theoretical predictions
- Test edge cases

### Reporting
When reporting results, include:
- Parameter values
- Random seed
- Environment (Python version, OS)
- Full output logs

## 🚫 What NOT to Contribute

- **Proprietary code**: All contributions must be open source
- **Unverified claims**: Always include supporting evidence
- **Breaking changes**: Maintain backward compatibility
- **Large datasets**: Use external hosting (Zenodo, etc.)

## 🎯 Priority Areas

We especially welcome contributions in:

1. **V29 ref_amp experiments**: Reproducing and extending V29 results
2. **AI order verification**: Independent replication of Appendix X §4
3. **Performance optimization**: Faster simulation for large-scale runs
4. **Educational materials**: Tutorials, videos, interactive demos
5. **Cross-validation**: Comparing V16f-η with other cosmological codes

## 📧 Questions?

- **Scientific questions**: Open a GitHub Discussion
- **Bug reports**: Open a GitHub Issue
- **General inquiries**: Email yagc-project@example.com

## 🙏 Acknowledgment

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in associated publications (if substantial)
- Invited to YAGC collaboration meetings

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all.

### Expected Behavior

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy toward other community members

### Unacceptable Behavior

- Harassment or discriminatory language
- Personal attacks
- Publishing others' private information
- Other conduct inappropriate in a professional setting

## 🌌 The Spirit of YAGC

Remember: This project emerged from dialogue between humans and AI. We welcome:
- Unconventional ideas
- Cross-disciplinary insights
- Computational experiments
- Philosophical reflections

The universe breathes — and so does science. Let's explore together.

---

**Thank you for contributing to YAGC!** 🚀
