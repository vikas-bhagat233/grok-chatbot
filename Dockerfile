# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install specific system dependencies for matplotlib/pillow if needed (usually fine on slim)
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose port 8000
EXPOSE 8000

# Define environment variable (Default, can be overridden)
# ENV GROQ_API_KEY=your_key_here 

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
