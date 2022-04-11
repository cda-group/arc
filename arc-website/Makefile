all:
	rm -r static/*
	make -C ../arc-docs build-with-deps; mv -f ../arc-docs/target/html static/docs
	make -C ../arc-report; mv -f ../arc-report/target/main.pdf static/Arc-Report.pdf
	zola build

build:
	zola build
