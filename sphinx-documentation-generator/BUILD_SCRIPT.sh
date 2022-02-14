rm -r ../docs/html
rm -r ../docs/doctrees
make html
mv ./_build/html ../docs/
mv ./_build/doctrees ../docs/
cd ../docs/

