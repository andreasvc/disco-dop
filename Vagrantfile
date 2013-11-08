# -*- mode: ruby -*-
# vi: set ft=ruby :

$script = <<SCRIPT
# install prerequisites
which cython || (
	apt-get update
	apt-get install -y build-essential python-dev python-numpy python-pip python-flask git
	pip install cython
)

# install disco-dop
which discodop || (
	cd /home/vagrant
	sudo -u vagrant git clone --depth 1 git://github.com/andreasvc/disco-dop.git
	cd disco-dop
	python setup.py install
)

# install tgrep2
which tgrep2 || (
	cd /home/vagrant
	sudo -u vagrant git clone --depth 1 git://github.com/andreasvc/tgrep2.git
	cd tgrep2
	sudo -u vagrant make && make install
)

# install style (part of diction)
which diction || (
	cd /home/vagrant
	sudo -u vagrant git clone --depth 1 git://github.com/andreasvc/diction.git
	cd diction
	sudo -u vagrant ./configure && sudo -u vagrant make && make install
)

# install alpinocorpus library
python -c 'import alpinocorpus' || (
	apt-get install -y python-software-properties
	add-apt-repository ppa:danieldk/dact
	apt-get update
	apt-get install -y libalpino-corpus2.0 libalpino-corpus-dev libxslt1-dev libxml2-dev libdbxml-dev libboost-all-dev
	sudo -u vagrant git clone https://github.com/andreasvc/alpinocorpus-python.git
	cd alpinocorpus-python
	sudo -u vagrant sed -i 's/-mt//g' setup.py
	sudo -u vagrant python setup.py config && sudo -u vagrant python setup.py build && python setup.py install
)
SCRIPT


# Vagrantfile API/syntax version. Don't touch unless you know what you're doing!
VAGRANTFILE_API_VERSION = "2"

Vagrant.configure(VAGRANTFILE_API_VERSION) do |config|
  # All Vagrant configuration is done here. The most common configuration
  # options are documented and commented below. For a complete reference,
  # please see the online documentation at vagrantup.com.

  # Every Vagrant virtual environment requires a box to build off of.
  config.vm.box = "precise64"

  # The url from where the 'config.vm.box' box will be fetched if it
  # doesn't already exist on the user's system.
  config.vm.box_url = "http://files.vagrantup.com/precise64.box"

  # Provisioning
  config.vm.provision :shell, :inline => $script

  # Create a forwarded port mapping which allows access to a specific port
  config.vm.network :forwarded_port, guest: 5000, host: 5000

  # Share an additional folder to the guest VM. The first argument is
  # the path on the host to the actual folder. The second argument is
  # the path on the guest to mount the folder. And the optional third
  # argument is a set of non-required options.
  # config.vm.synced_folder "../data", "/vagrant_data"

  # Provider-specific configuration so you can fine-tune various
  # backing providers for Vagrant. These expose provider-specific options.
  # Example for VirtualBox:
  #
  # config.vm.provider :virtualbox do |vb|
  #   # Don't boot with headless mode
  #   vb.gui = true
  #
  #   # Use VBoxManage to customize the VM. For example to change memory:
  #   vb.customize ["modifyvm", :id, "--memory", "1024"]
  # end
  #
  # View the documentation for the provider you're using for more
  # information on available options.
end
