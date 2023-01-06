for cfg in ./turnt-*.toml
do
    turnt -c $cfg ./bril/**/*.bril
done
