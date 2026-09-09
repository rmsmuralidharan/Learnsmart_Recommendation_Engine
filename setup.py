from setuptools import setup, find_packages

def get_requirements(file:str) -> list[str]:

    """
    Reads the requirements.txt and returns list of dependencies
    """

    requirements = []
    with open(file) as file_obj:
        requirements = file_obj.readlines()

        requirements = [req.replace('\n', '') for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')

            return requirements

        


setup(
    name="learnsmart-recommendation-engine",
    version= '0.1.0',
    description= 'LearnSmart AI is an intelligent Ed tech platform that uses machine learning to analyze student learning behaviour and recommend personlaized courses based on interests, expereince, learning style and performance',
    author="Muralidharan RMS",
    contact="rmsmuralidharan@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)