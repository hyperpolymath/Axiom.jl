# Governance

## Overview

Axiom.jl uses a **Benevolent Dictator for Life (BDFL) + Consensus** model, inspired by successful open source projects while incorporating RSR principles.

## Decision Making

### Levels of Decision

| Level | Scope | Process |
|-------|-------|---------|
| **Minor** | Bug fixes, docs, typos | Single maintainer approval |
| **Standard** | Features, refactors | Two maintainer approval |
| **Major** | Architecture, breaking changes | BDFL + community RFC |
| **Governance** | This document, CoC changes | BDFL + supermajority vote |

### RFC Process

For major changes:

1. **Draft RFC**: Open issue with `[RFC]` prefix
2. **Discussion Period**: Minimum 14 days
3. **Revision**: Address feedback
4. **Final Comment Period**: 7 days
5. **Decision**: BDFL decision with reasoning

## Roles

### BDFL (Benevolent Dictator for Life)

**Current BDFL**: [Project Founder]

Responsibilities:
- Final arbiter on technical disputes
- Guardian of project vision
- Emergency decisions when needed

The BDFL can be changed by unanimous agreement of Core Contributors.

### Core Contributors

Contributors with merge rights. Requirements:

- Sustained contributions over 6+ months
- Deep understanding of codebase
- Demonstrated alignment with project values
- Nominated by existing Core Contributor
- Approved by BDFL

**Current Core Contributors**: See `MAINTAINERS.md`

### Contributors

Anyone who has contributed code, docs, issues, or reviews.

### Community Members

Anyone participating in discussions, using the software, or providing feedback.

## Tri-Perimeter Contribution Framework (TPCF)

Following RSR standards, contributions are organized by trust level:

### 🔒 Perimeter 1 (Core)

**Access**: Core Contributors only

Areas:
- Build system (flake.nix, Justfile)
- CI/CD configuration
- Security-critical code
- FFI boundaries
- Release process

### 🧠 Perimeter 2 (Expert)

**Access**: Trusted Contributors (established track record)

Areas:
- Core algorithms
- Verification system
- Backend implementations
- API design
- Performance-critical code

### 🌱 Perimeter 3 (Community)

**Access**: Open to all

Areas:
- Documentation
- Examples
- Tests
- Bug reports
- Feature proposals
- Community support

## Meetings

### Technical Meetings

- **Frequency**: Monthly
- **Format**: Video call + text summary
- **Agenda**: Posted 7 days in advance
- **Notes**: Published in `docs/meetings/`

### Community Calls

- **Frequency**: Quarterly
- **Format**: Open video call
- **Purpose**: Community Q&A, roadmap discussion

## Conflict Resolution

1. **Discussion**: Try to resolve through discussion
2. **Mediation**: Involve neutral third party
3. **Escalation**: Bring to Core Contributors
4. **Final Decision**: BDFL makes final call

## Changes to Governance

This document can be changed through:

1. RFC process (standard)
2. 14-day discussion period
3. Supermajority (2/3) approval from Core Contributors
4. BDFL approval

## Code of Conduct

All participants must follow our [Code of Conduct](CODE_OF_CONDUCT.md).

Violations should be reported to: conduct@axiom-jl.org

## Licensing Decisions

- Core framework: MIT License
- All contributions must be MIT-compatible
- Third-party dependencies reviewed for license compatibility
- SPDX headers required on all source files

## Financial Transparency

- Funding sources listed in `FUNDING.yml`
- Financial reports published quarterly (when applicable)
- No individual may receive >50% of project funds
- All spending decisions made by Core Contributors

## Succession Planning

If the BDFL becomes unavailable:

1. Core Contributors elect interim leader (simple majority)
2. 90-day period to establish new governance
3. Options: New BDFL, Steering Committee, or Foundation

## Contact

- General: hello@axiom-jl.org
- Governance questions: governance@axiom-jl.org
- Security issues: security@axiom-jl.org

---

*This governance model follows RSR community governance standards.*
