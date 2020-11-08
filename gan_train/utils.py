import torch
import torch.nn as nn
import copy


def d_loop(generator, discriminator, d_optimizer, real_batch, fake_descriptors, cuda=False):
    criterion = nn.BCELoss()
    # 1. Train D on real+fake
    d_optimizer.zero_grad()

    #  1A: Train D on real
    d_real_data = real_batch
    if cuda:
        d_real_data = d_real_data.cuda()
    d_real_decision = discriminator(d_real_data)
    target = torch.ones_like(d_real_decision)
    if cuda:
        target = target.cuda()
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    d_gen_input = fake_descriptors
    if cuda:
        d_gen_input = d_gen_input.cuda()

    with torch.no_grad():
        d_fake_data = generator(d_gen_input)
    d_fake_decision = discriminator(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    if cuda:
        target = target.cuda()
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error
    d_loss.backward()
    d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
    return d_real_error.cpu().item(), d_fake_error.cpu().item()


def d_unrolled_loop(generator, discriminator, d_optimizer, real_batch, fake_descriptors, cuda=False):
    criterion = nn.BCELoss()
    # 1. Train D on real+fake
    d_optimizer.zero_grad()

    #  1A: Train D on real
    d_real_data = real_batch
    if cuda:
        d_real_data = d_real_data.cuda()
    d_real_decision = discriminator(d_real_data)
    target = torch.ones_like(d_real_decision)
    if cuda:
        target = target.cuda()
    d_real_error = criterion(d_real_decision, target)  # ones = true

    #  1B: Train D on fake
    if cuda:
        fake_descriptors = fake_descriptors.cuda()

    with torch.no_grad():
        d_fake_data = generator(fake_descriptors)
    d_fake_decision = discriminator(d_fake_data)
    target = torch.zeros_like(d_fake_decision)
    if cuda:
        target = target.cuda()
    d_fake_error = criterion(d_fake_decision, target)  # zeros = fake

    d_loss = d_real_error + d_fake_error
    d_loss.backward(create_graph=True)
    d_optimizer.step()  # Only optimizes D's parameters; changes based on stored gradients from backward()
    return d_real_error.cpu().item(), d_fake_error.cpu().item()


def g_loop(generator, discriminator, g_optimizer, d_optimizer, real_batch, fake_descriptors, cuda=False, unrolled_steps=10):
    criterion = nn.BCELoss()
    # 2. Train G on D's response (but DO NOT train D on these labels)
    g_optimizer.zero_grad()

    gen_input = fake_descriptors
    if cuda:
        gen_input = gen_input.cuda()

    if unrolled_steps > 0:
        backup = copy.deepcopy(discriminator)
        for i in range(unrolled_steps):
            d_unrolled_loop(generator=generator, discriminator=discriminator, d_optimizer=d_optimizer,
                            real_batch=real_batch, fake_descriptors=gen_input, cuda=cuda)
    g_fake_data = generator(gen_input)
    dg_fake_decision = discriminator(g_fake_data)
    target = torch.ones_like(dg_fake_decision)
    if cuda:
        target = target.cuda()
    g_error = criterion(dg_fake_decision, target)  # we want to fool, so pretend it's all genuine
    g_error.backward()
    g_optimizer.step()  # Only optimizes G's parameters

    if unrolled_steps > 0:
        discriminator.load(backup)
        del backup
    return g_error.cpu().item()


def g_sample(generator, fake_descriptors, cuda=False):
    with torch.no_grad():
        gen_input = fake_descriptors
        if cuda:
            gen_input = gen_input.cuda()
        g_fake_data = generator(gen_input)
        return g_fake_data.cpu().numpy()
