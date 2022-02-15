rm -r ../docs/*
>../docs/.nojekyll
make html
mv ./_build/html/* ../docs/
rm -r ./_build/

