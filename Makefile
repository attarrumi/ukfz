env:
	py -m venv env

run-env:
	env/Scripts/Activate.ps1


run-zig:
	zig build --watch run

build:
	zig build --release=fast