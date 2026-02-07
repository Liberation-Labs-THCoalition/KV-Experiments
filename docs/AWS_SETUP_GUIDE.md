# AWS Cloud Compute Setup Guide

Quick reference for setting up GPU compute for projector training experiments.

---

## Before We Start - Check Your Credits

1. Log into AWS Console: https://console.aws.amazon.com/
2. Go to **Billing Dashboard** (search "Billing" in top bar)
3. Look for **Credits** in the left sidebar
4. Note: How much credit? What services does it apply to?

Most free tier credits work with **SageMaker** or **EC2**. We'll use one of these.

---

## Option A: SageMaker Studio (Easier)

SageMaker is AWS's ML platform - more managed, less setup.

### Step 1: Open SageMaker
- Search "SageMaker" in AWS console
- Click **Amazon SageMaker**

### Step 2: Create a Domain (first time only)
- Click **Set up for single user** if prompted
- This creates a "Studio" environment
- Takes 5-10 minutes

### Step 3: Launch Studio
- Click **Studio** in left sidebar
- Click **Open Studio**

### Step 4: Start a GPU Notebook
- In Studio, create new notebook
- Choose instance type: **ml.g4dn.xlarge** (cheapest GPU, ~$0.50/hr)
  - Has NVIDIA T4 GPU (16GB VRAM)
  - Good enough for our projector training
- Or **ml.g5.xlarge** (~$1.00/hr) for more power

### Step 5: Upload Our Code
- We'll upload the projector training script
- Install dependencies in notebook

---

## Option B: EC2 Instance (More Control)

EC2 gives you a full virtual machine. More setup but more flexibility.

### Step 1: Launch EC2 Instance
- Search "EC2" in console
- Click **Launch Instance**

### Step 2: Choose AMI (Operating System)
- Search for **Deep Learning AMI**
- Select: "Deep Learning AMI GPU PyTorch 2.x (Ubuntu 22.04)"
- This comes with CUDA and PyTorch pre-installed

### Step 3: Choose Instance Type
- Filter by "GPU instances"
- **g4dn.xlarge** - 1x T4 GPU, 16GB VRAM (~$0.50/hr)
- **g5.xlarge** - 1x A10G GPU, 24GB VRAM (~$1.00/hr)

### Step 4: Key Pair
- Create new key pair (or use existing)
- Download .pem file - YOU NEED THIS TO CONNECT
- Save somewhere safe!

### Step 5: Security Group
- Allow SSH (port 22) from your IP
- Default settings usually work

### Step 6: Launch and Connect
```bash
# From Git Bash or terminal:
ssh -i "your-key.pem" ubuntu@<instance-public-ip>
```

---

## What We'll Need to Install

Once connected (either option), we need:

```bash
# If using EC2 Deep Learning AMI, most is pre-installed
# Just need our specific packages:

pip install transformers accelerate bitsandbytes
pip install datasets  # for training data

# Clone our repo or upload code
```

---

## Cost Estimates

| Instance | GPU | VRAM | $/hour | 10 hours |
|----------|-----|------|--------|----------|
| g4dn.xlarge | T4 | 16GB | $0.526 | $5.26 |
| g5.xlarge | A10G | 24GB | $1.006 | $10.06 |
| g5.2xlarge | A10G | 24GB | $1.212 | $12.12 |
| p3.2xlarge | V100 | 16GB | $3.06 | $30.60 |

**Recommendation:** Start with g4dn.xlarge. Cheap and sufficient for initial projector experiments.

---

## Important: Stop Instances When Done!

GPU instances charge BY THE HOUR even when idle.

- **SageMaker:** Stop notebook instance when not using
- **EC2:** Stop or terminate instance when done

Set a billing alarm:
1. Go to Billing → Budgets
2. Create budget → Cost budget
3. Set threshold (e.g., $20)
4. Add email alert

---

## Questions to Answer Before Setup

1. How much credit do you have?
2. Which region are your credits for? (us-east-1 is most common)
3. Do you have a preference: SageMaker (managed) vs EC2 (manual)?
4. Do you have an SSH client ready? (Git Bash works on Windows)

---

## When We Reconnect

We'll:
1. Review your credit situation
2. Pick SageMaker or EC2
3. Launch instance together
4. Upload projector training code
5. Run first experiment

---

*Don't worry about getting it perfect before we meet - just check your credits and we'll figure out the rest together.*
