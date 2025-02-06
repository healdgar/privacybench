I'm preparing to open source a "PET" project.  

A PET is a Privacy Enhancing Technology.



Here's the problem: 

- SaaS and Software businesses want to use usage data to improve their products.  They don't necessarily care about personal data within usage data, but, quite often there is no practical way to filter it out.



- Meanwhile the EU AI Act, GDPR, and AI Governance standards require a documented effort to minimize the amount of personal data used to only that, which is necessary.



Challenges:

- Traditional filtering methods look for hard-coded patterns (e.g.  Regex) to identify numbers that look like SSN, or credit card numbers, or other data, but cannot identify the pattern of a name, address, location, or combination of data points rendering a person identifiable.



Elements of a Solution:

- Leveraging LLM technology and its ability to understand the full context of data, essentially challenging the state-of-the-art to police the state-of-the-art.

- Deploying an LLM locally as a filter to 'sanitize' usage data to ensure all personal data is omitted so as not to accidentally breach personal data or have it pollute product insights.

- Establishing benchmarks to demonstrate trustworthiness of the technology.



Solution:

- Creation of PrivacyBench, a scoring method for LLM-based privacy applications.

- Demonstrating PrivacyBench for the use case of personal data redaction.

- Showing the market that local models can be proven effective enough to reliably detect, redact, and replace personal data, thereby protecting both consumers and businesses.

- Inspiring others to consider privacy applications as they train and design their LLMs to meet other, more general benchmarks.



Please have a look at my Github repo and contribute if you are interested!  Feel free to use anything you find (it's open-sourced under MIT), but don't forget to attribute!