
1. Clone the repository.
2. Move into the cloned repo.
3. To build the docker container execute: `docker build -f Dockerfile_diffusion -t diffusion-image .`
4. To start an container run: `docker run --rm -it -p 8890:8888 --name diffusion-tutorial diffusion-image`

Note that the container will be removed once terminated. Changes you make inside the container will be lost.
To make an persistent container remove the `--rm` flag, and potentially add storate through an mount point.

The passwort of the jupyter server is `boschai`.

