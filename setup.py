from setuptools import find_packages, setup

setup(
    name="ticketlens",
    version="0.1.0",
    description="TicketLens: Autonomous IT Ticket Classification & Triage Engine",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    python_requires=">=3.10",
)
